import os
import json
import time
import uuid
import httpx
import heapq
import pynvml
import logging
import asyncio
import statistics
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, HTTPException, Request, Response, status

from .core import health_monitor
from .core.metrics import metrics
from .core.config import load_config
from .core.manager import ModelManager
from .core.warmup import ModelWarmupManager
from .core.logging_server import setup_logging
from .core.health_monitor import HealthMonitor
from .core.errors import InsufficientVRAMError
from .core.limit_request import RequestSizeLimitMiddleware
from .core.telemetry import TelemetryCollector, RequestMetrics
from .core.model_status import ModelStatus, init_status_tracker
from .core.queue import QueueManager, QueuedRequest, RequestPriority, ModelRequestQueue
from .core.prometheus_metrics import (
    init_prometheus_collector, get_prometheus_collector, PrometheusMetricsCollector
)


# CONFIG_PATH bisa di-set via environment variable, default ke config.json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.getenv(
    "CONFIG_PATH", os.path.join(PROJECT_ROOT, "config.json"))
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

setup_logging(
    log_level=logging.INFO,
    use_structured=os.getenv("STRUCTURED_LOGS", "false").lower() == "true"
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app first
app = FastAPI()

try:
    _temp_config = load_config(CONFIG_PATH)

    # Setup middlewares yang butuh config
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_temp_config.api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
except Exception as e:
    logger.error(f"Failed to load config for middleware setup: {e}")

    # Default CORS jika config gagal
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Global variables yang akan di-init saat startup
config = None
manager = None
queue = None
warmup_manager = None
queue_manager = None
telemetry = None
http_client = None
gpu_handle = None
health_monitor = None
status_tracker = None
prometheus_collector = None


@app.on_event("startup")
async def app_startup():
    global config, manager, warmup_manager, queue_manager, telemetry, http_client, gpu_handle, health_monitor, status_tracker, prometheus_collector

    try:
        logger.info("Initializing ModelStatusTracker.")
        status_tracker = init_status_tracker()
        await status_tracker.set_server_status("initializing")

        logger.info(f"Loading config from: {CONFIG_PATH}")
        config = load_config(CONFIG_PATH)
        await status_tracker.initialize_from_config(list(config.models.keys()))

        logger.info("Initializing HTTP client.")
        limits = httpx.Limits(
            max_keepalive_connections=config.system.http_max_keepalive,
            max_connections=config.system.http_max_connections,
            keepalive_expiry=60.0
        )

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=2.0,  # Reduced from 10s - local connections are fast
                read=config.system.request_timeout_sec * 2,
                write=5.0,  # Reduced from 10s
                pool=5.0
            ),
            limits=limits,
            http2=False  # HTTP/1.1 faster for local llama.cpp connections
        )

        logger.info("Initializing ModelManager.")
        manager = ModelManager(config, shutdown_event)

        logger.info("Initializing QueueManager.")
        queue_manager = QueueManager(config)

        logger.info("Initializing WarmupManager.")
        warmup_manager = ModelWarmupManager(manager, config, shutdown_event)

        logger.info("Initializing TelemetryCollector.")
        telemetry = TelemetryCollector()

        logger.info("Initializing GPU monitoring.")
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        logger.info(
            f"Connected to GPU: {pynvml.nvmlDeviceGetName(gpu_handle)}")

        logger.info("Initializing health monitor.")
        health_monitor = HealthMonitor(manager, check_interval_sec=30)

        logger.info("Initializing Prometheus metrics collector.")
        prometheus_collector = init_prometheus_collector(
            gpu_device_index=config.system.gpu_devices[0]
        )

        logger.info("Starting warmup manager.")
        await warmup_manager.start()

        logger.info("Starting health monitoring.")
        health_monitor.start()

        logger.info("Starting model status sync task.")
        status_sync_task = asyncio.create_task(_sync_model_statuses())
        background_tasks.append(status_sync_task)

        await status_tracker.set_server_status("ready")

        logger.info("Server startup complete!")

    except Exception as e:
        logger.exception(f"FATAL: Gagal inisialisasi server: {e}")
        try:
            pynvml.nvmlShutdown()
        except:
            pass
        raise e


@app.on_event("shutdown")
async def app_shutdown():
    """Graceful shutdown dengan proper cleanup."""
    logger.info("Application shutdown initiated.")

    # Update status tracker
    if status_tracker:
        await status_tracker.set_server_status("shutting_down")

    # Set shutdown flag
    shutdown_event.set()

    # Cancel all background tasks first
    logger.info(f"Cancelling {len(background_tasks)} background tasks.")
    for task in background_tasks:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    # Stop health monitor
    if health_monitor:
        logger.info("Stopping health monitor.")
        try:
            await asyncio.wait_for(health_monitor.stop(), timeout=5.0)
            logger.info("Health monitor stopped")
        except asyncio.TimeoutError:
            logger.warning("Health monitor stop timeout")

    # Wait for active requests dengan timeout
    shutdown_timeout = 10
    start_time = time.time()

    while active_requests > 0 and (time.time() - start_time) < shutdown_timeout:
        logger.info(
            f"Waiting for {active_requests} active requests to complete.")
        await asyncio.sleep(1)

    if active_requests > 0:
        logger.warning(
            f"Shutdown timeout reached. Force closing with {active_requests} "
            f"requests still active."
        )

    # Stop all runners
    if manager:
        logger.info("Stopping all model runners.")
        try:
            await asyncio.wait_for(manager.stop_all_runners(), timeout=15.0)
            logger.info("All runners stopped")
        except asyncio.TimeoutError:
            logger.error("Timeout stopping runners. Force killing.")
            async with manager.lock:
                for runner in manager.active_runners.values():
                    if runner.process:
                        try:
                            runner.process.kill()
                            logger.warning(
                                f"Force killed runner: {runner.alias}")
                        except Exception as e:
                            logger.error(f"Error killing runner: {e}")

    # Close HTTP client
    if http_client:
        logger.info("Closing HTTP client.")
        try:
            await asyncio.wait_for(http_client.aclose(), timeout=5.0)
            logger.info("HTTP client closed")
        except asyncio.TimeoutError:
            logger.warning("HTTP client close timeout")

    # Shutdown NVML
    if gpu_handle:
        try:
            pynvml.nvmlShutdown()
            logger.info("NVML shutdown complete")
        except Exception as e:
            logger.error(f"NVML shutdown error: {e}")

    logger.info("Application shutdown complete")


class EjectRequest(BaseModel):
    model: str


# --- Event Handler FastAPI ---
shutdown_event = asyncio.Event()
active_requests = 0
active_requests_lock = asyncio.Lock()


# Track background tasks yang perlu di-cancel
background_tasks = []


async def _sync_model_statuses():
    """
    Background task untuk sync status model dari manager ke status tracker.

    Ini berjalan secara periodik untuk:
    1. Detect perubahan status runner
    2. Update VRAM usage per model
    3. Detect crashed/stopped runners
    """
    sync_interval = 2  # seconds

    logger.info("Model status sync task started")

    try:
        while not shutdown_event.is_set():
            try:
                # Wait dengan shutdown check
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=sync_interval
                    )
                    # Shutdown triggered
                    break
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue sync

                # Skip jika manager belum ready
                if manager is None or status_tracker is None:
                    continue

                # Sync setiap model
                async with manager.lock:
                    # Get all models from config
                    all_models = set(config.models.keys()) if config else set()
                    active_models = set(manager.active_runners.keys())

                    # Update status untuk active models
                    for alias, runner in manager.active_runners.items():
                        # Map runner status ke ModelStatus
                        runner_status = runner.status

                        if runner_status == "ready":
                            model_status = ModelStatus.READY
                        elif runner_status == "loading":
                            model_status = ModelStatus.LOADING
                        elif runner_status == "starting":
                            model_status = ModelStatus.STARTING
                        elif runner_status == "crashed":
                            model_status = ModelStatus.CRASHED
                        elif runner_status == "stopped":
                            model_status = ModelStatus.OFF
                        else:
                            model_status = ModelStatus.UNKNOWN

                        # Get VRAM usage jika ada
                        vram_mb = None
                        if alias in manager.vram_tracker.model_tracks:
                            track = manager.vram_tracker.model_tracks[alias]
                            vram_mb = track.current_vram_used_mb

                        # Update status tracker
                        await status_tracker.update_status(
                            alias=alias,
                            status=model_status,
                            port=runner.port if runner.is_alive() else None,
                            error_message=runner.startup_error if runner_status == "crashed" else None,
                            vram_used_mb=vram_mb
                        )

                    # Update models yang tidak active ke OFF
                    for alias in all_models - active_models:
                        # Check jika ada di failed_models
                        if alias in manager.failed_models:
                            failed_info = manager.failed_models[alias]
                            await status_tracker.update_status(
                                alias=alias,
                                status=ModelStatus.FAILED,
                                error_message=failed_info.get(
                                    "error", "Unknown error")
                            )
                        else:
                            # Model tidak aktif
                            current_status = await status_tracker.get_status(alias)
                            if current_status and current_status.status not in [ModelStatus.OFF, ModelStatus.FAILED]:
                                await status_tracker.update_status(
                                    alias=alias,
                                    status=ModelStatus.OFF
                                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in status sync: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    except asyncio.CancelledError:
        logger.info("Model status sync task cancelled")
    except Exception as e:
        logger.exception(f"Fatal error in status sync task: {e}")
    finally:
        logger.info("Model status sync task stopped")


# --- Custom fungsi ---
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware untuk track active requests."""
    global active_requests

    # Jika sedang shutdown, reject new requests
    if shutdown_event.is_set():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Server is shutting down"}
        )

    async with active_requests_lock:
        active_requests += 1

    try:
        response = await call_next(request)
        return response
    finally:
        async with active_requests_lock:
            active_requests -= 1


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware untuk collect metrics."""
    start_time = time.time()

    try:
        response = await call_next(request)

        # Record success
        endpoint = request.url.path
        metrics["requests_total"][endpoint] += 1

        if response.status_code < 400:
            metrics["requests_success"][endpoint] += 1
        else:
            metrics["requests_failed"][endpoint] += 1

        duration = time.time() - start_time
        metrics["request_duration_seconds"][endpoint].append(duration)

        # Keep only last 1000 durations per endpoint
        if len(metrics["request_duration_seconds"][endpoint]) > 1000:
            metrics["request_duration_seconds"][endpoint] = \
                metrics["request_duration_seconds"][endpoint][-1000:]

        return response

    except Exception as e:
        endpoint = request.url.path
        metrics["requests_total"][endpoint] += 1
        metrics["requests_failed"][endpoint] += 1
        raise


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """Middleware untuk collect telemetry dan prometheus metrics."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    request.state.request_id = request_id
    request.state.start_time = start_time
    request.state.tokens_generated = 0

    # Skip monitoring endpoints
    skip_endpoints = [
        '/health', '/metrics', '/metrics/stream', '/metrics/report',
        '/v1/telemetry/summary', '/vram', '/v1/health/models'
    ]

    if request.url.path in skip_endpoints:
        return await call_next(request)

    # Get prometheus collector
    prom_collector = get_prometheus_collector()
    model_alias = None

    try:
        response = await call_next(request)

        # Get model_alias dengan fallback
        model_alias = getattr(request.state, 'model_alias', None)
        if model_alias is None:
            model_alias = "unknown"

        end_time = time.time()
        duration = end_time - start_time
        tokens = getattr(request.state, 'tokens_generated', 0)
        queue_time = getattr(request.state, 'queue_time', 0.0)

        # Determine status
        status = "success" if response.status_code < 400 else "error"

        # Record to existing telemetry
        metrics_data = RequestMetrics(
            request_id=request_id,
            model_alias=model_alias,
            endpoint=request.url.path,
            start_time=start_time,
            end_time=end_time,
            status_code=response.status_code,
            queue_time=queue_time,
            processing_time=getattr(request.state, 'processing_time', 0.0),
            tokens_generated=tokens
        )
        await telemetry.record_request(metrics_data)

        # Record to prometheus collector
        if prom_collector and model_alias != "unknown":
            await prom_collector.record_request_end(
                model=model_alias,
                endpoint=request.url.path,
                duration_seconds=duration,
                status=status,
                tokens=tokens,
                queue_wait_seconds=queue_time,
                status_code=response.status_code
            )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        # Handle error cases
        model_alias = getattr(request.state, 'model_alias', 'unknown')
        end_time = time.time()
        duration = end_time - start_time

        # Record to existing telemetry
        metrics_data = RequestMetrics(
            request_id=request_id,
            model_alias=model_alias,
            endpoint=request.url.path,
            start_time=start_time,
            end_time=end_time,
            error=str(e)
        )
        await telemetry.record_request(metrics_data)

        # Record to prometheus collector
        if prom_collector and model_alias != "unknown":
            await prom_collector.record_request_end(
                model=model_alias,
                endpoint=request.url.path,
                duration_seconds=duration,
                status="error",
                tokens=0,
                queue_wait_seconds=0,
                status_code=500
            )

        raise


async def _process_queued_request(
    queued_req_data: Dict[str, Any],
    runner,
    endpoint: str
) -> Dict[str, Any]:
    """Process single queued request with retry logic."""
    body = queued_req_data["body"]
    request_id = queued_req_data["request_id"]
    model_alias = queued_req_data["model_alias"]

    start_time = time.time()
    max_retries = 2
    retry_delay = 1.0

    logger.debug(
        f"[Queue] Processing request {request_id} for {model_alias}")

    for attempt in range(max_retries + 1):
        try:
            # Build request
            internal_url = f"{runner.url}{endpoint}"
            req = http_client.build_request(
                method="POST",
                url=internal_url,
                json=body,
                headers={"Content-Type": "application/json"}
            )

            build_time = time.time() - start_time

            # Send request
            send_start = time.time()
            response_stream = await http_client.send(req, stream=True)
            send_time = time.time() - send_start

            # Check if streaming
            is_streaming = body.get("stream", False)

            if is_streaming:
                # For streaming, collect all chunks
                chunks = []
                async for chunk in response_stream.aiter_bytes():
                    chunks.append(chunk)
                await response_stream.aclose()

                total_time = time.time() - start_time
                logger.debug(
                    f"[Queue] Request {request_id} timing: build={build_time:.3f}s, "
                    f"send={send_time:.3f}s, total={total_time:.3f}s (streaming)"
                )

                # Return raw chunks for client streaming
                return {
                    "type": "stream",
                    "chunks": chunks,
                    "status_code": response_stream.status_code
                }
            else:
                # Non-streaming: read full response
                read_start = time.time()
                content = await response_stream.aread()
                read_time = time.time() - read_start
                await response_stream.aclose()

                # Parse response
                response_data = json.loads(content.decode('utf-8'))

                # Detect context shift warning
                context_shifted = False
                if 'timings' in response_data:
                    timings = response_data['timings']
                    # llama.cpp mengirim context_shift di timings
                    if 'context_shift' in timings and timings['context_shift'] > 0:
                        context_shifted = True
                        logger.warning(
                            f"[{model_alias}] Context shift detected! "
                            f"Shifted {timings['context_shift']} tokens. "
                            f"Consider using shorter conversations or higher n_ctx."
                        )

                # Extract tokens
                tokens = 0
                if 'usage' in response_data:
                    tokens = response_data['usage'].get('completion_tokens', 0)
                elif 'choices' in response_data and response_data['choices']:
                    content_text = response_data['choices'][0].get(
                        'message', {}).get('content', '')
                    tokens = len(content_text) // 4

                # Inject context shift warning ke response jika terjadi
                if context_shifted:
                    if 'choices' in response_data and response_data['choices']:
                        # Tambah warning di metadata response
                        if 'metadata' not in response_data:
                            response_data['metadata'] = {}
                        response_data['metadata']['context_shifted'] = True
                        response_data['metadata']['warning'] = "Context window exceeded. Some earlier messages were shifted out. Consider using shorter conversations."

                total_time = time.time() - start_time
                logger.info(
                    f"[Queue] Request {request_id} timing: build={build_time:.3f}s, "
                    f"send={send_time:.3f}s, read={read_time:.3f}s, total={total_time:.3f}s"
                )

                return {
                    "type": "json",
                    "data": response_data,
                    "tokens": tokens,
                    "status_code": response_stream.status_code,
                    "context_shifted": context_shifted
                }

        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            # Retriable errors
            if attempt < max_retries:
                logger.warning(
                    f"[Queue] Request {request_id} attempt {attempt + 1}/{max_retries + 1} "
                    f"failed with {type(e).__name__}: {e}. Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                logger.error(
                    f"[Queue] Request {request_id} failed after {max_retries + 1} attempts: {e}"
                )
                raise
        except Exception as e:
            logger.exception(
                f"[Queue] Error processing request {request_id}: {e}")
            raise

    # Should never reach here
    raise RuntimeError(
        f"Unexpected end of retry loop for request {request_id}")


async def _process_request_via_queue(
    queue: ModelRequestQueue,
    request_id: str,
    model_alias: str,
    body: Dict[str, Any],
    priority: RequestPriority,
    endpoint: str,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Process request via queue system.

    This function:
    1. Enqueues request
    2. Waits for queue processor to handle it
    3. Returns result

    Note: Runner is obtained by the queue processor, not here,
    to avoid double runner acquisition.
    """

    # Use provided timeout or default from config with multiplier
    if timeout is None:
        timeout = config.system.queue_timeout_sec * 3

    enqueue_start = time.time()

    # Create future for this request
    response_future = asyncio.Future()

    # Enqueue request first
    queued_req = QueuedRequest(
        priority=priority.value,
        timestamp=time.time(),
        request_id=request_id,
        model_alias=model_alias,
        body=body,
        response_future=response_future
    )

    async with queue.lock:
        if len(queue.queue) >= queue.max_queue_size:
            queue.total_rejected += 1
            # Track queue rejection to Prometheus
            prom_collector = get_prometheus_collector()
            if prom_collector:
                prom_collector.record_queue_rejected(model_alias)
            raise RuntimeError(
                f"Queue for model '{model_alias}' is full ({queue.max_queue_size}). "
            )

        # Use heapq for O(log n) insertion instead of O(n) manual insertion
        heapq.heappush(queue.queue, queued_req)

        queue.total_requests += 1

        # Update queue depth in Prometheus
        prom_collector = get_prometheus_collector()
        if prom_collector:
            prom_collector.update_queue_depth(model_alias, len(queue.queue))

        # Signal that queue has items
        queue.queue_not_empty.set()

        # Start queue processor if not running
        should_start_processor = not queue.processing

    # Start processor outside the lock to avoid deadlock
    if should_start_processor:
        logger.info(
            f"[Queue] Starting processor for {model_alias} (was stopped, auto-restarting)")
        asyncio.create_task(_queue_processor(model_alias, queue, endpoint))

    # Wait for result with timeout
    try:
        result = await asyncio.wait_for(response_future, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        wait_time = time.time() - enqueue_start
        logger.error(
            f"[Queue] Request {request_id} timeout after {wait_time:.1f}s (timeout={timeout}s, queue_length={len(queue.queue)})"
        )
        # Remove from queue if timeout
        async with queue.lock:
            try:
                queue.queue.remove(queued_req)
            except ValueError:
                pass  # Already processed
        raise TimeoutError(f"Request timeout after {timeout}s in queue")


async def _queue_processor(model_alias: str, queue: ModelRequestQueue, endpoint: str):
    """
    Background task to process queue for a specific model.

    This handles:
    - Dequeuing requests in priority order
    - Sending to llama-server
    - Handling responses
    - Error recovery
    """
    async with queue.lock:
        if queue.processing:
            return  # Already processing
        queue.processing = True

    logger.info(f"[Queue] Starting processor for model '{model_alias}'")

    idle_timeout = config.system.queue_processor_idle_sec  # Configurable timeout

    try:
        while True:
            # Try to dequeue immediately first
            queued_req = await queue.dequeue()

            if queued_req is None:
                # Queue empty, clear the event and wait for signal or timeout
                queue.queue_not_empty.clear()

                try:
                    await asyncio.wait_for(
                        queue.queue_not_empty.wait(),
                        timeout=idle_timeout
                    )
                    # Don't clear here - let the next iteration handle it
                    continue
                except asyncio.TimeoutError:
                    # Check if this is a warm model before stopping
                    is_warm = warmup_manager.is_model_warm(model_alias)

                    if is_warm:
                        # Warm model - DON'T stop processor, keep running
                        logger.debug(
                            f"[Queue] Processor idle for {model_alias}, but it's a warm model. Keeping processor alive."
                        )
                        continue
                    else:
                        # Not warm - safe to stop
                        logger.info(
                            f"[Queue] Processor idle for {model_alias} (not warm), stopping"
                        )
                        break

            # Increment processing counter
            async with queue.lock:
                queue.current_processing += 1

            # Process request
            try:
                process_start = time.time()
                logger.info(
                    f"[Queue] Processing request {queued_req.request_id} "
                    f"for {model_alias} (priority: {queued_req.priority})"
                )

                # Get runner
                runner_start = time.time()
                runner = await manager.get_runner_for_request(model_alias)
                runner_time = time.time() - runner_start

                if runner_time > 0.1:  # Log if getting runner takes more than 100ms
                    logger.warning(
                        f"[Queue] Slow runner acquisition for {queued_req.request_id}: {runner_time:.3f}s"
                    )

                # Process the request
                request_start = time.time()
                result = await _process_queued_request(
                    {
                        "body": queued_req.body,
                        "request_id": queued_req.request_id,
                        "model_alias": model_alias
                    },
                    runner,
                    endpoint
                )
                request_time = time.time() - request_start

                # Set result to future
                if not queued_req.response_future.done():
                    queued_req.response_future.set_result(result)

                async with queue.lock:
                    queue.total_processed += 1

                total_process_time = time.time() - process_start
                queue_wait_time = process_start - queued_req.timestamp

                logger.info(
                    f"[Queue] Request {queued_req.request_id} completed: "
                    f"queue_wait={queue_wait_time:.3f}s, runner={runner_time:.3f}s, "
                    f"request={request_time:.3f}s, total={total_process_time:.3f}s"
                )

            except Exception as e:
                logger.exception(
                    f"[Queue] Error processing request {queued_req.request_id}: {e}"
                )

                # Set exception to future
                if not queued_req.response_future.done():
                    queued_req.response_future.set_exception(e)

            finally:
                # Decrement processing counter
                async with queue.lock:
                    queue.current_processing -= 1

    except asyncio.CancelledError:
        logger.info(f"[Queue] Processor cancelled for {model_alias}")
        raise
    except Exception as e:
        logger.exception(f"[Queue] Processor error for {model_alias}: {e}")
    finally:
        async with queue.lock:
            queue.processing = False
        logger.info(f"[Queue] Processor stopped for model '{model_alias}'")


async def _proxy_request_with_queue(request: Request, endpoint: str):
    """
    Enhanced proxy with queue system.

    Flow:
    1. Parse request and validate
    2. Add to queue dengan priority
    3. Queue processor akan handle request
    4. Return response (streaming atau json)
    """
    try:
        # Parse body
        body = await request.json()
        model_alias = body.get("model")

        if not model_alias:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'model' wajib ada di JSON body."
            )

        # Set model alias for telemetry
        request.state.model_alias = model_alias

        # Check if streaming
        is_streaming = body.get("stream", False)

        # Determine priority from header (optional)
        priority_header = request.headers.get(
            "X-Request-Priority", "normal").lower()
        priority_map = {
            "high": RequestPriority.HIGH,
            "normal": RequestPriority.NORMAL,
            "low": RequestPriority.LOW
        }
        priority = priority_map.get(priority_header, RequestPriority.NORMAL)

        # Record request for warmup
        warmup_manager.record_request(model_alias)

        # Get queue for this model
        queue = await queue_manager.get_queue(model_alias)

        # Generate request ID
        request_id = request.state.request_id  # From telemetry middleware

        # Queue the request
        queue_start_time = time.time()

        logger.info(
            f"[Queue] Enqueuing request {request_id} for {model_alias} "
            f"(priority: {priority.name}, streaming: {is_streaming})"
        )

        try:
            # Process request through queue with extended timeout
            # Use 3x config timeout for high-latency workloads
            queue_timeout = config.system.queue_timeout_sec * 3

            result = await _process_request_via_queue(
                queue=queue,
                request_id=request_id,
                model_alias=model_alias,
                body=body,
                priority=priority,
                endpoint=endpoint,
                timeout=queue_timeout
            )

            # Record queue time
            request.state.queue_time = time.time() - queue_start_time

            # Handle response based on type
            if result["type"] == "stream":
                # Streaming response
                async def stream_generator():
                    try:
                        for chunk in result["chunks"]:
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error in stream generator: {e}")
                        error_chunk = f'data: {{"error": "Stream error"}}\n\n'
                        yield error_chunk.encode()

                return StreamingResponse(
                    stream_generator(),
                    status_code=result["status_code"],
                    media_type="text/event-stream"
                )
            else:
                # JSON response
                request.state.tokens_generated = result.get("tokens", 0)

                return JSONResponse(
                    content=result["data"],
                    status_code=result["status_code"]
                )

        except TimeoutError as e:
            queue_wait_time = time.time() - queue_start_time
            logger.error(
                f"[Queue] Request {request_id} timeout after {queue_wait_time:.1f}s: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timeout in queue after {queue_wait_time:.1f}s: {str(e)}"
            )
        except InsufficientVRAMError as e:
            # VRAM not enough to load model
            logger.warning(
                f"[Queue] Insufficient VRAM for {model_alias}: "
                f"need {e.required_mb:.0f} MB, have {e.available_mb:.0f} MB"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "insufficient_vram_error",
                        "code": "vram_exhausted",
                        "model": e.model_alias,
                        "required_mb": round(e.required_mb),
                        "available_mb": round(e.available_mb),
                        "loaded_models": e.loaded_models
                    }
                }
            )
        except RuntimeError as e:
            # Queue full or other runtime errors
            error_msg = str(e)
            if "queue" in error_msg.lower():
                logger.warning(f"[Queue] Queue full for {model_alias}: {e}")
            else:
                logger.warning(f"[Queue] Runtime error for {model_alias}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )

    except HTTPException:
        raise
    except LookupError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unexpected error in proxy_request_with_queue")

        if DEBUG_MODE:
            detail = f"Internal Server Error: {str(e)}"
        else:
            detail = "Internal server error occurred."

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


async def _proxy_embeddings(request: Request):
    """
    Embeddings endpoint dengan format OpenAI-compatible.
    """
    try:
        # Baca body request
        body = await request.json()
        model_alias = body.get("model")

        if not model_alias:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'model' wajib ada di JSON body."
            )

        # Validate model alias format
        if not all(c.isalnum() or c in ('-', '_') for c in model_alias):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model alias hanya boleh mengandung alphanumeric, dash, dan underscore."
            )

        # Set model_alias untuk telemetry
        request.state.model_alias = model_alias

        # Verify model supports embeddings
        model_config = config.models.get(model_alias)
        if not model_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_alias}' tidak ditemukan di config."
            )

        if not model_config.params.embedding:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_alias}' tidak mendukung embeddings. Set 'embedding: true' di config."
            )

        # Get input text(s)
        input_data = body.get("input")
        if not input_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'input' wajib ada untuk embeddings."
            )

        # Normalize input to list
        if isinstance(input_data, str):
            inputs = [input_data]
        elif isinstance(input_data, list):
            inputs = input_data
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'input' harus berupa string atau list of strings."
            )

        # Record request untuk warmup
        warmup_manager.record_request(model_alias)

        # Get runner
        queue_start = time.time()
        runner = await manager.get_runner_for_request(model_alias)

        # Build request untuk llama-server embedding endpoint
        internal_url = f"{runner.url}/embedding"

        # Collect all embeddings
        all_embeddings = []
        total_tokens = 0

        for idx, text in enumerate(inputs):
            # llama-server expects { "content": "text" }
            embed_body = {"content": text}

            req = http_client.build_request(
                method="POST",
                url=internal_url,
                json=embed_body,
                headers={"Content-Type": "application/json"}
            )

            response = await http_client.send(req)

            if response.status_code != 200:
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Embedding request failed: {error_detail}"
                )

            result = response.json()

            # llama-server returns list directly, not dict with "embedding" key
            if isinstance(result, list):
                embedding = result
            elif isinstance(result, dict):
                embedding = result.get("embedding", [])
            else:
                logger.error(
                    f"Unexpected embedding response format: {type(result)}")
                embedding = []

            all_embeddings.append({
                "object": "embedding",
                "embedding": embedding,
                "index": idx
            })

            # Estimate tokens (rough: ~1 token per 4 chars)
            total_tokens += len(text) // 4

        request.state.queue_time = time.time() - queue_start
        request.state.tokens_generated = 0  # Embeddings don't generate tokens

        # Return OpenAI-compatible format
        return JSONResponse(content={
            "object": "list",
            "data": all_embeddings,
            "model": model_alias,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        })

    except LookupError as e:
        logger.error(f"Model tidak ditemukan: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

    except httpx.ConnectError as e:
        logger.warning(f"Error koneksi untuk {model_alias}: {e}.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Runner untuk '{model_alias}' tidak tersedia."
        )

    except httpx.TimeoutException as e:
        logger.error(f"Timeout untuk {model_alias}: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timeout untuk model '{model_alias}'"
        )

    except RuntimeError as e:
        # Untuk error seperti max concurrent models
        logger.error(f"Runtime error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )

    except Exception as e:
        logger.exception("Terjadi error tidak terduga di proxy_request")

        if DEBUG_MODE:
            detail = f"Internal Server Error: {str(e)}"
        else:
            detail = "Ada yang error."

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


# --- Endpoint API ---
@app.get("/ping")
def serverless_ping_check():
    """For the sake of serverless to detect if the server is ready"""
    return {'status': 'ok'}



@app.get("/health")
def health_check():
    """Mengecek apakah API Gateway hidup dan semua dependencies OK."""
    try:
        health_status = {
            "status": "ok",
            "checks": {}
        }

        # Check GPU
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            health_status["checks"]["gpu"] = {
                "status": "ok",
                "vram_used_gb": f"{mem_info.used / (1024**3):.2f}"
            }
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["checks"]["gpu"] = {
                "status": "error",
                "error": str(e)
            }

        # Check manager
        try:
            active_count = len([r for r in manager.active_runners.values()
                                if r.is_alive()])
            health_status["checks"]["manager"] = {
                "status": "ok",
                "active_models": active_count
            }
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["checks"]["manager"] = {
                "status": "error",
                "error": str(e)
            }

        # Check http_client
        try:
            if http_client.is_closed:
                health_status["status"] = "degraded"
                health_status["checks"]["http_client"] = {
                    "status": "error",
                    "error": "HTTP client is closed"
                }
            else:
                health_status["checks"]["http_client"] = {
                    "status": "ok"
                }
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["checks"]["http_client"] = {
                "status": "error",
                "error": str(e)
            }

        # Jika ada component yang error, return 503
        if health_status["status"] == "degraded":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )

        return health_status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check gagal: {e}")


@app.get("/metrics/legacy")
async def get_metrics_legacy():
    """Legacy metrics endpoint (old format). Use /metrics for Prometheus format."""
    output = []

    # Request metrics
    for endpoint, count in metrics["requests_total"].items():
        output.append(f'requests_total{{endpoint="{endpoint}"}} {count}')
        output.append(
            f'requests_success{{endpoint="{endpoint}"}} {metrics["requests_success"][endpoint]}')
        output.append(
            f'requests_failed{{endpoint="{endpoint}"}} {metrics["requests_failed"][endpoint]}')

        # Duration statistics
        durations = metrics["request_duration_seconds"].get(endpoint, [])
        if durations:
            output.append(
                f'request_duration_seconds_avg{{endpoint="{endpoint}"}} {statistics.mean(durations):.4f}')
            # quantiles requires at least 2 data points
            if len(durations) >= 2:
                output.append(
                    f'request_duration_seconds_p95{{endpoint="{endpoint}"}} {statistics.quantiles(durations, n=20)[18]:.4f}')

    # Model metrics
    output.append(f'models_loaded_total {metrics["models_loaded_total"]}')
    output.append(f'models_ejected_total {metrics["models_ejected_total"]}')
    output.append(
        f'models_active {len([r for r in manager.active_runners.values() if r.is_alive()])}')

    # VRAM metrics
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        output.append(f'vram_total_bytes {mem_info.total}')
        output.append(f'vram_used_bytes {mem_info.used}')
        output.append(f'vram_free_bytes {mem_info.free}')
    except:
        pass

    return Response(content="\n".join(output), media_type="text/plain")


@app.get("/v1/telemetry/summary")
async def get_telemetry_summary():
    """Get telemetry summary."""
    return telemetry.get_summary()


@app.get("/v1/health/models")
async def get_models_health():
    """Get health status untuk semua active models."""
    return health_monitor.get_all_health()


# --- Prometheus Metrics Endpoints ---

@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus exposition format metrics endpoint.

    This endpoint is designed to be scraped by Prometheus server.
    Returns all metrics in Prometheus text format.

    Usage:
        - Configure Prometheus to scrape this endpoint
        - Default scrape interval: 15s
    """
    collector = get_prometheus_collector()
    if not collector:
        return Response(
            content="# Prometheus collector not initialized\n",
            media_type="text/plain",
            status_code=503
        )

    content = collector.get_prometheus_metrics()
    return Response(
        content=content,
        media_type=collector.get_content_type()
    )


@app.get("/metrics/stream")
async def stream_metrics(request: Request):
    """
    Server-Sent Events (SSE) endpoint for real-time metrics streaming.

    Streams metrics updates every 2 seconds with per-model breakdown.

    Event types:
        - metrics: Real-time metrics snapshot
        - heartbeat: Keep-alive every 30 seconds

    Usage:
        curl -N http://localhost:8000/metrics/stream
    """
    collector = get_prometheus_collector()
    if not collector:
        return JSONResponse(
            {"error": "Prometheus collector not initialized"},
            status_code=503
        )

    async def event_generator():
        heartbeat_interval = 30
        metrics_interval = 2
        last_heartbeat = time.time()

        try:
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    break

                # Check shutdown
                if shutdown_event.is_set():
                    yield f"event: shutdown\ndata: {{\"message\": \"Server shutting down\"}}\n\n"
                    break

                # Send metrics
                try:
                    snapshot = await collector.get_realtime_snapshot(
                        manager=manager,
                        queue_manager=queue_manager
                    )
                    yield f"event: metrics\ndata: {json.dumps(snapshot)}\n\n"
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(
                        f"Error generating metrics snapshot: {error_msg}")
                    # Send error event so client knows what happened
                    yield f"event: error\ndata: {{\"error\": \"{error_msg}\"}}\n\n"

                # Send heartbeat if needed (only after 30 seconds)
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f"event: heartbeat\ndata: {{\"timestamp\": \"{datetime.now().isoformat()}\"}}\n\n"
                    last_heartbeat = current_time

                await asyncio.sleep(metrics_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in metrics stream: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/metrics/report")
async def get_metrics_report():
    """
    Get detailed 5-minute aggregated metrics report.

    Returns comprehensive metrics including:
        - Server status and uptime
        - GPU VRAM usage (MB and GB)
        - Per-model detailed breakdown:
            - Request counts (total, success, errors)
            - Latency statistics (avg, min, max, p95)
            - Queue statistics
            - Token throughput
        - Aggregated totals

    Usage:
        curl http://localhost:8000/metrics/report
    """
    collector = get_prometheus_collector()
    if not collector:
        return JSONResponse(
            {"error": "Prometheus collector not initialized"},
            status_code=503
        )

    try:
        report = await collector.get_5min_report(
            manager=manager,
            queue_manager=queue_manager
        )
        return report
    except Exception as e:
        logger.error(f"Error generating metrics report: {e}")
        return JSONResponse(
            {"error": f"Failed to generate report: {str(e)}"},
            status_code=500
        )


# --- Model Status Realtime Endpoints ---


@app.get("/v1/models/status")
async def get_all_models_status():
    """
    Get status semua model secara lengkap.

    Returns:
        - server: Status server (initializing/ready/shutting_down)
        - models: Dict semua model dengan statusnya
        - summary: Ringkasan jumlah model per status

    Status yang mungkin:
        - off: Model tidak aktif
        - starting: Subprocess sedang di-spawn
        - loading: Model sedang di-load ke VRAM
        - ready/loaded: Model siap menerima request
        - stopping: Model sedang dihentikan
        - crashed: Model crash
        - failed: Model gagal start
    """
    if not status_tracker:
        # Fallback: baca dari file jika tracker belum ready
        from .core.model_status import ModelStatusTracker
        file_status = ModelStatusTracker.read_status_file()
        if file_status:
            return file_status
        return {"error": "Status tracker not initialized", "models": {}}

    return await status_tracker.get_full_status()


@app.get("/v1/models/status/stream")
async def stream_models_status(request: Request):
    """
    SSE endpoint untuk realtime status updates.

    Event types:
        - full_status: Status lengkap saat pertama connect
        - model_update: Update status satu model
        - server_update: Update status server
        - heartbeat: Keep-alive setiap 30 detik
    """
    if not status_tracker:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Status tracker not initialized"}
        )

    async def event_generator():
        # Subscribe ke updates
        queue = await status_tracker.subscribe()

        try:
            # Kirim full status saat pertama connect
            full_status = await status_tracker.get_full_status()
            yield f"event: full_status\ndata: {json.dumps(full_status)}\n\n"

            heartbeat_interval = 30  # seconds
            last_heartbeat = time.time()

            while True:
                # Check jika client disconnect
                if await request.is_disconnected():
                    break

                try:
                    # Wait for update dengan timeout untuk heartbeat
                    try:
                        update = await asyncio.wait_for(
                            queue.get(),
                            timeout=heartbeat_interval
                        )

                        # Send update
                        event_type = update.get("type", "update")
                        yield f"event: {event_type}\ndata: {json.dumps(update.get('data', update))}\n\n"

                    except asyncio.TimeoutError:
                        # Send heartbeat
                        heartbeat_data = {
                            "timestamp": datetime.now().isoformat(),
                            "server_status": status_tracker.server_status
                        }
                        yield f"event: heartbeat\ndata: {json.dumps(heartbeat_data)}\n\n"
                        last_heartbeat = time.time()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    break

        finally:
            # Unsubscribe saat disconnect
            await status_tracker.unsubscribe(queue)
            logger.debug("SSE client disconnected")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/vram")
def get_vram_status():
    """
    Memantau VRAM secara dinamis dengan tracking per model.

    Returns:
        - GPU info (total, used, free)
        - Per-model VRAM usage
        - Can load more models
        - Status (healthy/warning/critical)
    """
    try:
        # Get comprehensive VRAM report dari tracker
        vram_report = manager.vram_tracker.get_vram_report()

        return vram_report

    except Exception as e:
        logger.exception(f"Error getting VRAM status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal membaca info VRAM: {e}"
        )


@app.get("/vram/models/{model_alias}")
async def get_model_vram_detail(model_alias: str):
    """
    Get detailed VRAM info untuk specific model.

    Args:
        model_alias: Alias model yang ingin di-check

    Returns:
        Detailed VRAM usage, snapshots, dan load history
    """
    try:
        async with manager.vram_tracker.lock:
            if model_alias not in manager.vram_tracker.model_tracks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_alias}' tidak ditemukan dalam VRAM tracking."
                )

            track = manager.vram_tracker.model_tracks[model_alias]

            # Get current VRAM info
            vram_info = manager.vram_tracker.get_current_vram_info()

            # Build detailed response
            return {
                "model_alias": model_alias,
                "port": track.port,
                "status": track.status,
                "vram_usage": {
                    "current_mb": round(track.current_vram_used_mb, 2),
                    "current_gb": round(track.current_vram_used_mb / 1024, 2),
                    "average_mb": round(track.get_average_usage_mb(), 2),
                    "percentage_of_total": round(
                        (track.current_vram_used_mb /
                         vram_info["total_mb"]) * 100, 2
                    ) if vram_info["total_mb"] > 0 else 0
                },
                "load_info": {
                    "start_time": track.load_start_time.isoformat() if track.load_start_time else None,
                    "end_time": track.load_end_time.isoformat() if track.load_end_time else None,
                    "duration_sec": track.get_load_duration_sec(),
                    "initial_free_vram_mb": round(track.initial_vram_free_mb, 2)
                },
                "snapshots": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "vram_used_mb": round(s.vram_used_mb, 2),
                        "status": s.status
                    }
                    for s in track.snapshots
                ],
                "current_gpu_state": {
                    "total_mb": round(vram_info["total_mb"], 2),
                    "used_mb": round(vram_info["used_mb"], 2),
                    "free_mb": round(vram_info["free_mb"], 2)
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting VRAM detail for {model_alias}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal membaca detail VRAM: {e}"
        )


@app.get("/vram/summary")
def get_vram_summary():
    """
    Get ringkasan VRAM yang simpel dan cepat.

    Returns:
        Ringkasan singkat status VRAM (good untuk monitoring dashboard)
    """
    try:
        vram_report = manager.vram_tracker.get_vram_report()

        return {
            "status": vram_report["status"],
            "total_gb": vram_report["gpu_info"]["total_gb"],
            "used_gb": vram_report["gpu_info"]["used_gb"],
            "free_gb": vram_report["gpu_info"]["free_gb"],
            "usage_percentage": vram_report["gpu_info"]["usage_percentage"],
            "loaded_models": vram_report["loaded_models_count"],
            "can_load_more": vram_report["can_load_more"],
            "models": [
                {
                    "alias": m["model_alias"],
                    "vram_gb": m["vram_used_gb"],
                    "percentage": m.get("vram_percentage", 0)
                }
                for m in vram_report["models"]
                if m["status"] == "loaded"
            ]
        }

    except Exception as e:
        logger.exception(f"Error getting VRAM summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal membaca summary VRAM: {e}"
        )


# --- Endpoint OpenAI-Compatible ---
@app.get("/v1/models")
def get_models():
    """Mengembalikan daftar model yang tersedia dari config.json."""
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "owned_by": "user",
                "n_ctx": conf.params.n_ctx
            }
            for alias, conf in config.models.items()
        ]
    }


@app.get("/v1/queue/stats")
async def get_queue_stats():
    """
    Get detailed statistics for all model queues.

    Returns:
    - Queue length (current pending requests)
    - Total requests processed
    - Rejection count (queue full)
    - Current processing count
    - Processing status
    """
    stats = queue_manager.get_all_stats()

    # Add summary
    total_queued = sum(q["queue_length"] for q in stats.values())
    total_processing = sum(q["current_processing"] for q in stats.values())
    total_processed = sum(q["total_processed"] for q in stats.values())
    total_rejected = sum(q["total_rejected"] for q in stats.values())

    return {
        "summary": {
            "total_queued": total_queued,
            "total_processing": total_processing,
            "total_processed": total_processed,
            "total_rejected": total_rejected
        },
        "per_model": stats
    }


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Chat completions dengan request queue system.

    Supports:
    - Priority queueing (via X-Request-Priority header: high/normal/low)
    - Streaming and non-streaming responses
    - Backpressure control
    - Fair scheduling

    Headers:
    - X-Request-Priority: high|normal|low (optional, default: normal)
    """
    return await _proxy_request_with_queue(request, "/v1/chat/completions")


@app.post("/v1/embeddings")
async def proxy_embeddings(request: Request):
    """
    Embeddings.
    """
    return await _proxy_embeddings(request)


@app.post("/v1/models/eject")
async def eject_model(request: EjectRequest):
    """
    (Eject) model yang sedang berjalan.
    """
    try:
        success = await manager.eject_model(request.model)
        if success:
            return {
                "status": "success",
                "model_ejected": request.model,
                "message": f"Model '{request.model}' berhasil dihentikan"
            }
        else:
            return {
                "status": "not_found",
                "model_ejected": None,
                "message": f"Model '{request.model}' tidak sedang berjalan."
            }
    except Exception as e:
        logger.exception(f"Gagal eject model {request.model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal eject model: {e}")


@app.get("/v1/models/{model_alias}/status")
async def get_model_loading_status(model_alias: str):
    try:
        status_info = await manager.get_model_status(model_alias)
        return {"model": model_alias, **status_info}
    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Gagal mendapatkan status untuk {model_alias}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal mendapatkan status: {e}")


@app.post("/v1/models/{model_alias}/reset")
async def reset_model_failure(model_alias: str):
    """
    Reset failed model status to allow retry.

    Useful when you've fixed configuration and want to retry.
    """
    try:
        async with manager.lock:
            if model_alias in manager.failed_models:
                failed_info = manager.failed_models[model_alias]
                del manager.failed_models[model_alias]

                return {
                    "status": "success",
                    "model": model_alias,
                    "message": f"Model failure status cleared. Had {failed_info['attempts']} failed attempts.",
                    "previous_error": failed_info['error']
                }
            else:
                return {
                    "status": "not_found",
                    "model": model_alias,
                    "message": f"Model '{model_alias}' has no failure record."
                }

    except Exception as e:
        logger.exception(f"Error resetting model {model_alias}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset model: {e}"
        )


@app.get("/v1/models/failed")
async def get_failed_models():
    """Get list of models that have failed to start."""
    async with manager.lock:
        if not manager.failed_models:
            return {
                "failed_models": [],
                "message": "No failed models"
            }

        return {
            "failed_models": [
                {
                    "model": alias,
                    "attempts": info["attempts"],
                    "error": info["error"][:200]
                }
                for alias, info in manager.failed_models.items()
            ]
        }
