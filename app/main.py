import os
import time
import uuid
import httpx
import pynvml
import logging
import asyncio
import statistics
from typing import Any, Dict
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
from .core.limit_request import RequestSizeLimitMiddleware
from .core.telemetry import TelemetryCollector, RequestMetrics
from .core.queue import QueueManager, QueuedRequest, RequestPriority, ModelRequestQueue

# Get absolute path to config.json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configOriginal.json")
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


@app.on_event("startup")
async def app_startup():
    """Startup tasks - Initialize semua dependencies."""
    global config, manager, warmup_manager, queue_manager, telemetry, http_client, gpu_handle, health_monitor

    try:
        logger.info(f"Loading config from: {CONFIG_PATH}")
        config = load_config(CONFIG_PATH)

        logger.info("Initializing HTTP client.")
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=config.system.request_timeout_sec,
                write=10.0,
                pool=5.0
            )
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

        logger.info("Starting warmup manager.")
        await warmup_manager.start()

        logger.info("Starting health monitoring.")
        health_monitor.start()

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
    """Middleware untuk collect telemetry."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    request.state.request_id = request_id
    request.state.start_time = start_time
    request.state.tokens_generated = 0

    try:
        response = await call_next(request)

        # Get model_alias dengan fallback
        model_alias = getattr(request.state, 'model_alias', None)

        # Jika tidak ada model_alias dan bukan monitoring endpoint, skip telemetry
        if model_alias is None:
            # Skip telemetry untuk monitoring endpoints
            if request.url.path in ['/health', '/metrics', '/v1/telemetry/summary', '/vram', '/v1/health/models']:
                return response
            model_alias = "unknown"

        metrics_data = RequestMetrics(
            request_id=request_id,
            model_alias=model_alias,
            endpoint=request.url.path,
            start_time=start_time,
            end_time=time.time(),
            status_code=response.status_code,
            queue_time=getattr(request.state, 'queue_time', 0.0),
            processing_time=getattr(request.state, 'processing_time', 0.0),
            tokens_generated=getattr(request.state, 'tokens_generated', 0)
        )

        await telemetry.record_request(metrics_data)
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        # Handle error cases
        model_alias = getattr(request.state, 'model_alias', 'unknown')
        metrics_data = RequestMetrics(
            request_id=request_id,
            model_alias=model_alias,
            endpoint=request.url.path,
            start_time=start_time,
            end_time=time.time(),
            error=str(e)
        )
        await telemetry.record_request(metrics_data)
        raise


async def _process_queued_request(
    queued_req_data: Dict[str, Any],
    runner,
    endpoint: str
) -> Dict[str, Any]:
    """Process single queued request."""
    body = queued_req_data["body"]
    request_id = queued_req_data["request_id"]

    logger.debug(
        f"[Queue] Processing request {request_id} for {queued_req_data['model_alias']}")

    try:
        # Build request
        internal_url = f"{runner.url}{endpoint}"
        req = http_client.build_request(
            method="POST",
            url=internal_url,
            json=body,
            headers={"Content-Type": "application/json"}
        )

        # Send request
        response_stream = await http_client.send(req, stream=True)

        # Check if streaming
        is_streaming = body.get("stream", False)

        if is_streaming:
            # For streaming, collect all chunks
            chunks = []
            async for chunk in response_stream.aiter_bytes():
                chunks.append(chunk)
            await response_stream.aclose()

            # Return raw chunks for client streaming
            return {
                "type": "stream",
                "chunks": chunks,
                "status_code": response_stream.status_code
            }
        else:
            # Non-streaming: read full response
            content = await response_stream.aread()
            await response_stream.aclose()

            # Parse response
            import json
            response_data = json.loads(content.decode('utf-8'))

            # Extract tokens
            tokens = 0
            if 'usage' in response_data:
                tokens = response_data['usage'].get('completion_tokens', 0)
            elif 'choices' in response_data and response_data['choices']:
                content_text = response_data['choices'][0].get(
                    'message', {}).get('content', '')
                tokens = len(content_text) // 4

            return {
                "type": "json",
                "data": response_data,
                "tokens": tokens,
                "status_code": response_stream.status_code
            }

    except Exception as e:
        logger.exception(f"[Queue] Error processing request {request_id}: {e}")
        raise


async def _process_request_via_queue(
    queue: ModelRequestQueue,
    request_id: str,
    model_alias: str,
    body: Dict[str, Any],
    priority: RequestPriority,
    endpoint: str
) -> Dict[str, Any]:
    """
    Process request via queue system.

    This function:
    1. Ensures runner is available
    2. Enqueues request
    3. Waits for queue processor to handle it
    4. Returns result
    """

    # Ensure runner is started (cold start if needed)
    runner = await manager.get_runner_for_request(model_alias)

    # Create future for this request
    response_future = asyncio.Future()

    # Start queue processor if not running
    if not queue.processing:
        asyncio.create_task(_queue_processor(model_alias, queue, endpoint))

    # Enqueue request
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
            raise RuntimeError(
                f"Queue for model '{model_alias}' is full ({queue.max_queue_size}). "
                f"Try again later."
            )

        # Insert with priority
        inserted = False
        for i, existing_req in enumerate(queue.queue):
            if queued_req.sort_key < existing_req.sort_key:
                queue.queue.insert(i, queued_req)
                inserted = True
                break

        if not inserted:
            queue.queue.append(queued_req)

        queue.total_requests += 1
        queue.queue_not_empty.set()

    # Wait for result with timeout
    timeout = config.system.queue_timeout_sec
    try:
        result = await asyncio.wait_for(response_future, timeout=timeout)
        return result
    except asyncio.TimeoutError:
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

    try:
        while True:
            # Dequeue next request
            queued_req = await queue.dequeue()

            if queued_req is None:
                # Queue empty, wait for signal or timeout
                try:
                    await asyncio.wait_for(
                        queue.queue_not_empty.wait(),
                        timeout=30  # 30 seconds idle timeout
                    )
                    queue.queue_not_empty.clear()
                    continue
                except asyncio.TimeoutError:
                    # No requests in 30s, stop processor
                    logger.info(
                        f"[Queue] Processor idle for {model_alias}, stopping")
                    break

            # Increment processing counter
            async with queue.lock:
                queue.current_processing += 1

            # Process request
            try:
                logger.info(
                    f"[Queue] Processing request {queued_req.request_id} "
                    f"for {model_alias} (priority: {queued_req.priority})"
                )

                # Get runner
                runner = await manager.get_runner_for_request(model_alias)

                # Process the request
                result = await _process_queued_request(
                    {
                        "body": queued_req.body,
                        "request_id": queued_req.request_id,
                        "model_alias": model_alias
                    },
                    runner,
                    endpoint
                )

                # Set result to future
                if not queued_req.response_future.done():
                    queued_req.response_future.set_result(result)

                async with queue.lock:
                    queue.total_processed += 1

                logger.debug(
                    f"[Queue] Request {queued_req.request_id} completed successfully"
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

        # Validate model alias
        if not model_alias.replace('-', '').replace('_', '').isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model alias hanya boleh mengandung alphanumeric, dash, dan underscore."
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
            # Process request through queue
            result = await _process_request_via_queue(
                queue=queue,
                request_id=request_id,
                model_alias=model_alias,
                body=body,
                priority=priority,
                endpoint=endpoint
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
            logger.error(f"[Queue] Request {request_id} timeout: {e}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timeout in queue: {str(e)}"
            )
        except RuntimeError as e:
            # Queue full
            logger.warning(f"[Queue] Queue full for {model_alias}: {e}")
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


@app.get("/metrics")
async def get_metrics():
    """Endpoint untuk metrics (Prometheus-compatible text format)."""
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


@app.get("/vram")
def get_vram_status():
    """
    Memantau VRAM.
    """
    global gpu_handle

    try:
        # Re-check GPU handle untuk memastikan GPU masih available
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        except pynvml.NVMLError as e:
            # Jika GPU error, coba re-init
            logger.warning(
                f"GPU error detected: {e}. Trying to re-initialize.")
            try:
                pynvml.nvmlShutdown()
                pynvml.nvmlInit()
                gpu_handle_new = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_handle = gpu_handle_new
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            except Exception as reinit_error:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"GPU tidak tersedia: {reinit_error}"
                )

        # Runner yang aktif dari manager
        active_runners_info = {}
        for alias, runner in manager.active_runners.items():
            if runner and runner.is_alive():
                active_runners_info[alias] = {
                    "port": runner.port,
                    "status": runner.status
                }

        return {
            "vram": {
                "total_gb": f"{mem_info.total / (1024**3):.2f}",
                "used_gb": f"{mem_info.used / (1024**3):.2f}",
                "free_gb": f"{mem_info.free / (1024**3):.2f}"
            },
            "active_models": active_runners_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal membaca info VRAM: {e}")


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
