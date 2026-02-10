"""
Model Manager - llama-server Process Management

Modul ini merupakan core dari router model yang mengelola lifecycle
llama-server processes untuk setiap model.

Components:
    - RunnerProcess: Wrapper untuk subprocess llama-server
    - ModelManager: Manager untuk semua active runners

RunnerProcess Responsibilities:
    - Start llama-server subprocess dengan konfigurasi yang sesuai
    - Health check polling sampai model ready
    - Graceful shutdown dengan escalating termination (SIGTERM -> SIGKILL)
    - Logging ke file per-runner

ModelManager Responsibilities:
    - Port allocation dari pool (8085-8584)
    - VRAM estimation dan tracking via VRAMTracker
    - Concurrent model limiting
    - Idle timeout watchdog (auto-unload model idle)
    - Retry logic untuk failed loads

llama-server Command Building:
    Router otomatis membangun command llama-server dengan flags:
    - --model: Path ke file GGUF
    - --host/--port: Internal binding
    - --n-gpu-layers, --ctx-size, --batch-size: Dari config
    - --parallel, --threads: Dari system config
    - -fa: Flash attention
    - --embedding: Jika model embedding
    - --no-context-shift: Untuk model non-SWA (auto-detect)
    - --cache-type-k/v: KV cache quantization

Request Flow:
    1. Request masuk ke ModelManager.get_runner_for_request()
    2. Check apakah model sudah running
    3. Jika belum, estimate VRAM dan check availability
    4. Start RunnerProcess dan wait for ready
    5. Return runner URL untuk request forwarding

VRAM Management:
    - Sequential loading via load_lock (prevent race conditions)
    - Pre-load VRAM estimation: file_size * multiplier + KV cache + overhead
    - Post-load actual tracking (before/after delta)
    - Reject load jika VRAM insufficient

Usage:
    manager = ModelManager(config, shutdown_event)
    runner = await manager.get_runner_for_request("qwen3-8b")
    # runner.url = "http://127.0.0.1:8085"
    
    # Stop specific model
    await manager.eject_model("qwen3-8b")
    
    # Stop all
    await manager.stop_all_runners()
"""

import gc
import os
import time
import httpx
import signal
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .metrics import metrics
from .vram_tracker import VRAMTracker
from .errors import InsufficientVRAMError
from .config import AppConfig, ModelConfig
from .gguf_utils import get_optimal_parallel, get_model_info
from .prometheus_metrics import get_prometheus_collector

logger = logging.getLogger(__name__)


class RunnerProcess:
    # Global cache: {model_path: (parallel, model_info)}
    _gguf_cache: Dict[str, tuple] = {}

    def __init__(self, alias: str, config: ModelConfig, port: int, llama_server_path: str, system_config):
        self.alias = alias
        self.config = config
        self.port = port
        self.llama_server_path = llama_server_path
        self.system_config = system_config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.last_used_time = time.time()
        self.started_time: Optional[float] = None  # Track start via time
        self.url = f"http://127.0.0.1:{self.port}"
        self.startup_error: Optional[str] = None
        self.status: str = "stopped"

        # Buat dan taruh log
        self.log_dir = Path("logs/runners")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{alias}_{port}.log"

        # Retry jika crash
        self.retry_count = 0
        self.max_retries = system_config.model_load_max_retries

    def is_alive(self) -> bool:
        if self.process is None:
            self.status = "stopped"
            return False

        is_running = self.process.returncode is None

        if not is_running:
            self.status = "crashed"

        return is_running

    async def start(self):
        if self.is_alive():
            logger.warning(
                f"[{self.alias}] Proses sudah berjalan.")
            return

        # Track start time
        self.started_time = time.time()
        self.status = "starting"

        params = self.config.params
        model_path = self.config.get_resolved_path()  # Use resolved path
        parallel = self.system_config.parallel_requests or 2

        if model_path in self._gguf_cache:
            parallel, model_info = self._gguf_cache[model_path]
            logger.debug(f"[{self.alias}] Using cached GGUF metadata")
        else:
            # Determine optimal parallel setting using GGUF metadata
            # base_parallel = params.parallel_override if params.parallel_override else self.system_config.parallel_requests

            parallel = self.system_config.parallel_requests
            # parallel, parallel_reason = get_optimal_parallel(
            #     model_path=model_path,
            #     n_ctx=params.n_ctx,
            #     default_parallel=base_parallel,
            #     min_ctx_per_slot=2048
            # )

            # if parallel != base_parallel:
            #     logger.info(f"[{self.alias}] {parallel_reason}")

        # Get model info
        model_info = get_model_info(model_path)

        # Cache it
        self._gguf_cache[model_path] = (parallel, model_info)

        # Store model_info untuk digunakan di command building
        self.model_info = model_info

        # Log model info for debugging
        if model_info:
            swa_status = f"SWA={model_info.swa_window_size}" if model_info.is_swa else "non-SWA"
            logger.info(
                f"[{self.alias}] Model: {model_info.name} | Arch: {model_info.architecture} | SWA Status: {swa_status} | Layers: {model_info.block_count} | Parallel: {parallel}"
            )

        command = [
            self.llama_server_path, "--model", self.config.get_resolved_path(),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--n-gpu-layers", str(params.n_gpu_layers),
            "--ctx-size", str(params.n_ctx), "--mlock", "--jinja"
        ]

        # Context shifting: DISABLE untuk non-SWA models, ENABLE untuk SWA models
        # SWA models BUTUH context shifting untuk handle long conversations
        if model_info and not model_info.is_swa:
            command.append("--no-context-shift")
            logger.info(
                f"[{self.alias}] Non-SWA model detected. Context shifting DISABLED.")
        else:
            logger.info(
                f"[{self.alias}] SWA model detected. Context shifting ENABLED for long conversations.")

        # RoPE frequency base (hanya jika di-set, biarkan default model jika None)
        if params.rope_freq_base is not None and params.rope_freq_base > 0:
            command.extend(["--rope-freq-base", str(params.rope_freq_base)])

        # Batch size (with per-model override)
        batch_size = params.batch_override if params.batch_override else params.n_batch
        command.extend(["--batch-size", str(batch_size)])

        # Parallel requests (already adjusted above for SWA models)
        command.extend(["--parallel", str(parallel)])

        # CPU threads
        command.extend(["--threads", str(self.system_config.cpu_threads)])

        # Flash Attention
        command.extend(["-fa", self.system_config.flash_attention])
        # if self.system_config.flash_attention == 'on' or 'auto':
        #     command.extend(['-fa'])

        # Memory mapping (conditional)
        if not self.system_config.use_mmap:
            command.append("--no-mmap")

        # Embedding mode
        if params.embedding:
            command.append("--embedding")

        # Chat template
        if params.chat_template:
            command.extend(["--chat-template", params.chat_template])

        # Cache types (only add if not None and not empty string)
        if params.type_k and params.type_k.lower() != "none":
            command.extend(["--cache-type-k", params.type_k])

        if params.type_v and params.type_v.lower() != "none":
            command.extend(["--cache-type-v", params.type_v])

        model_size_gb = Path(self.config.get_resolved_path()
                             ).stat().st_size / (1024**3)
        
        if params.additional_parameter:
            logger.info('Injecting additional raw parameters')
            command.extend([params.additional_parameter])

        if model_info and model_info.block_count > 20:
            logger.info(
                f"[{self.alias}] Large model detected ({model_size_gb:.1f} GB). Performing memory cleanup."
            )

            # Force garbage collection
            gc.collect()

            # Small delay to allow system to stabilize
            await asyncio.sleep(0.5)

            logger.info(
                f"[{self.alias}] Memory cleanup complete. Starting load.")

        self.startup_error = None

        # Buka log file untuk stdout dan stderr
        log_handle = open(self.log_file, 'w')
        self.log_handle = log_handle  # Store early for cleanup in stop()

        try:
            logger.info(f'Running {self.config.get_resolved_path()} - command [{command}]')
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log_handle,
                stderr=subprocess.STDOUT
            )

            # Track waktu mulai subprocess
            subprocess_start = time.time()

            # Tunggu sampai subprocess benar-benar start
            subprocess_time = time.time() - subprocess_start

            # Tunggu sampai siap via health check
            health_check_start = time.time()
            await self._wait_for_ready()
            health_check_time = time.time() - health_check_start
            total_startup_time = time.time() - self.started_time

            self.last_used_time = time.time()
            self.status = "ready"

            logger.info(
                f"[{self.alias}] READY at {self.url} | Total: {total_startup_time:.2f}s | (subprocess: {subprocess_time:.2f}s, loading: {health_check_time:.2f}s)"
            )
        except Exception as e:
            # Close log handle on exception to prevent file descriptor leak
            if log_handle and not log_handle.closed:
                try:
                    log_handle.close()
                except:
                    pass
            raise  # Re-raise the exception

    async def stop(self):
        """
        Stop the runner process gracefully with escalating termination strategy.

        Strategy:
        1. SIGTERM (graceful) - wait up to 15 seconds
        2. SIGKILL (force) - if SIGTERM times out
        3. os.kill() as last resort for processes if models are still alive
        """
        if not self.is_alive() or self.process is None:
            self.status = "stopped"
            logger.debug(f"[{self.alias}] Process already stopped or None.")
            return

        pid = self.process.pid
        logger.info(
            f"[{self.alias}] Menghentikan proses (Port {self.port}, PID {pid}).")

        try:
            # Step 1: Try graceful termination with SIGTERM
            self.process.terminate()
            logger.debug(
                f"[{self.alias}] Sent SIGTERM, waiting for graceful shutdown...")

            try:
                await asyncio.wait_for(self.process.wait(), timeout=15.0)
                logger.info(f"[{self.alias}] Berhasil dihentikan (graceful).")
            except asyncio.TimeoutError:
                # Step 2: SIGTERM timeout, escalate to SIGKILL via asyncio
                logger.warning(
                    f"[{self.alias}] SIGTERM timeout (15s). Escalating to SIGKILL...")

                try:
                    self.process.kill()
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    logger.info(
                        f"[{self.alias}] Berhasil dihentikan (force kill).")
                except asyncio.TimeoutError:
                    # Step 3: Last resort - use os.kill directly
                    logger.warning(
                        f"[{self.alias}] asyncio SIGKILL timeout. Trying os.kill() directly on PID {pid}...")

                    try:
                        os.kill(pid, signal.SIGKILL)
                        # Give it a moment
                        await asyncio.sleep(1.0)

                        # Check if process is really dead
                        try:
                            # Signal 0 = check if process exists
                            os.kill(pid, 0)
                            logger.error(
                                f"[{self.alias}] Process PID {pid} masih hidup setelah os.kill(). Mungkin zombie atau kernel issue."
                            )
                        except OSError:
                            # Process is dead (OSError means process doesn't exist)
                            logger.info(
                                f"[{self.alias}] Berhasil dihentikan via os.kill().")
                    except ProcessLookupError:
                        logger.info(
                            f"[{self.alias}] Process sudah mati sebelum os.kill().")
                    except Exception as e:
                        logger.error(
                            f"[{self.alias}] os.kill() error: {e}")

                except Exception as e:
                    logger.error(f"[{self.alias}] Error during SIGKILL: {e}")

        except ProcessLookupError:
            # Process already dead
            logger.info(
                f"[{self.alias}] Process sudah tidak ada (already dead).")
        except Exception as e:
            logger.error(f"[{self.alias}] Unexpected error during stop: {e}")
        finally:
            # Always cleanup
            self.process = None
            self.status = "stopped"

            # Close log file handle
            if hasattr(self, 'log_handle') and self.log_handle:
                try:
                    self.log_handle.close()
                except Exception as e:
                    logger.debug(
                        f"[{self.alias}] Error closing log handle: {e}")

    async def _wait_for_ready(self, timeout=120):
        self.status = "loading"
        start_time = time.time()
        last_log_time = 0

        # Optimized adaptive polling intervals - faster initial checks
        # Small models (like gemma3-270m) can be ready in <1s, so we poll aggressively
        poll_intervals = [
            0.02,   # 20ms - very fast initial checks for quick models
            0.02,
            0.05,   # 50ms
            0.05,
            0.1,    # 100ms - transition
            0.1,
            0.2,    # 200ms
            0.3,    # 300ms
            0.5,    # 500ms - for slower loads
            0.5,
        ]
        poll_index = 0
        max_fast_polls = len(poll_intervals)

        async with httpx.AsyncClient() as client:
            iteration = 0
            while time.time() - start_time < timeout:
                if not self.is_alive():
                    try:
                        with open(self.log_file, 'r') as f:
                            lines = f.readlines()
                            self.startup_error = ''.join(lines)
                    except Exception as e:
                        self.startup_error = f"Process crashed, cannot read log: {e}"

                    self.status = "crashed"
                    logger.error(
                        f"[{self.alias}] Crashed. Error: {self.startup_error}...")
                    raise Exception(
                        f"Failed to start model. Error: {self.startup_error}...")

                try:
                    # Use shorter timeout for health check
                    response = await client.get(f"{self.url}/health", timeout=0.5)

                    if response.status_code == 200:  # Ready
                        elapsed = time.time() - start_time
                        self.status = "ready"
                        logger.info(
                            f"[{self.alias}] READY in {elapsed:.2f}s (after {iteration} health checks)")
                        return

                    elif response.status_code == 503:  # Model loading
                        current_time = time.time()

                        if current_time - last_log_time >= 3.0:  # Log progress every 3 seconds
                            elapsed = current_time - start_time
                            logger.info(
                                f"[{self.alias}] Loading... ({elapsed:.1f}s elapsed, status 503)")
                            last_log_time = current_time

                        self.status = "loading"

                    else:
                        logger.warning(
                            f"[{self.alias}] Unexpected status: {response.status_code}")

                except httpx.ConnectError:  # Connection refused
                    current_time = time.time()

                    if current_time - last_log_time >= 5.0:
                        elapsed = current_time - start_time
                        logger.debug(
                            f"[{self.alias}] Waiting for server to start... ({elapsed:.1f}s)")
                        last_log_time = current_time

                    self.status = "starting"

                except httpx.TimeoutException:  # Health check timeout
                    logger.debug(
                        f"[{self.alias}] Health check timeout")

                # Adaptive sleep - fast at first, slower later
                # MUST be inside while loop for proper polling
                if poll_index < max_fast_polls:
                    sleep_time = poll_intervals[poll_index]
                    poll_index += 1
                else:
                    # After fast polling phase, use 1 second interval
                    sleep_time = 1.0

                await asyncio.sleep(sleep_time)
                iteration += 1

        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(
            f"[{self.alias}] Failed to start after {elapsed:.1f}s (timeout: {timeout}s)")

        await self.stop()
        self.status = "crashed"
        raise TimeoutError(
            f"Runner {self.alias} failed to start within {timeout} seconds.")


class ModelManager:
    def __init__(self, config: AppConfig, shutdown_event):
        self.config = config
        self.shutdown_event = shutdown_event
        self.active_runners: Dict[str, RunnerProcess] = {}
        self.port_pool = set(range(8085, 8585))
        self.used_ports = set()
        self.lock = asyncio.Lock()
        self.gpu_devices = config.system.gpu_devices

        # {model_alias: {error: str, attempts: int}}
        self.failed_models: Dict[str, Dict] = {}

        # VRAM Tracker initialization
        self.vram_tracker = VRAMTracker(
            gpu_device_index=self.gpu_devices[0],
            min_vram_required=config.system.min_vram_required
        )
        self.vram_tracker.start_monitoring()
        logger.info("VRAM Tracker initialized and monitoring started")

        self.check_task = asyncio.create_task(self._idle_check_watchdog())

    def _allocate_port(self) -> int:
        """Allocate port dari pool yang available."""
        available_ports = self.port_pool - self.used_ports
        if not available_ports:
            raise RuntimeError(
                "Tidak ada port tersedia. Semua port sudah digunakan.")

        port = min(available_ports)  # Ambil port terkecil yang available
        self.used_ports.add(port)
        return port

    def _release_port(self, port: int):
        """Release port kembali ke pool."""
        if port in self.used_ports:
            self.used_ports.remove(port)

    async def _idle_check_watchdog(self):
        timeout = self.config.system.idle_timeout_sec
        max_time = 300
        timeout_enabled = self.config.system.enable_idle_timeout

        if not timeout_enabled:
            logger.info(
                "[Idle Watchdog] Idle timeout DISABLED. Models akan tetap loaded di VRAM.")

        try:
            while not self.shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=60
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                current_time = time.time()
                runners_to_stop = []

                async with self.lock:
                    for alias, runner in list(self.active_runners.items()):
                        # Check shutdown
                        if self.shutdown_event.is_set():
                            return

                        # Skip jika runner sudah mati - akan di-cleanup saat request berikutnya
                        if not runner.is_alive():
                            logger.debug(
                                f"[Idle Watchdog] Skipping dead runner '{alias}'"
                            )
                            continue

                        # Jika stuck di loading terlalu lama, stop
                        if runner.status in ["loading", "starting"]:
                            if runner.started_time and (current_time - runner.started_time) > max_time:
                                logger.warning(
                                    f"Model '{alias}' stuck di status '{runner.status}' lebih dari {max_time}s. Forcing stop."
                                )
                                runners_to_stop.append((alias, runner.port))
                            continue

                        # Check idle timeout untuk model yang ready (hanya jika enabled)
                        if runner.status == "ready":
                            if timeout_enabled:
                                idle_time = current_time - runner.last_used_time
                                if idle_time > timeout:
                                    logger.info(
                                        f"Model '{alias}' idle selama {idle_time:.0f}s (>{timeout}s). Stopping..."
                                    )
                                    runners_to_stop.append(
                                        (alias, runner.port))
                            # else: skip timeout check, model tetap loaded

                # Stop runners di luar lock untuk menghindari deadlock
                for alias, port in runners_to_stop:
                    async with self.lock:
                        runner = self.active_runners.get(alias)
                        if runner and runner.port == port:  # Pastikan masih runner yang sama
                            await runner.stop()
                            self._release_port(port)
                            del self.active_runners[alias]

                            # Juga hapus dari VRAM tracker
                            await self.vram_tracker.track_model_eject(alias)

        except asyncio.CancelledError:
            logger.info("Idle check watchdog cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in idle check watchdog: {e}")

    async def get_runner_for_request(self, model_alias: str) -> RunnerProcess:
        if model_alias not in self.config.models:
            raise LookupError(
                f"Model '{model_alias}' tidak terdefinisi di config.json.")

        if model_alias in self.failed_models:
            failed_info = self.failed_models[model_alias]
            if failed_info["attempts"] >= 3:  # Max global attempts
                raise RuntimeError(
                    f"Model '{model_alias}' has failed {failed_info['attempts']} times. Last error: {failed_info['error']}..."
                )

        # Menyimpan runner
        runner: Optional[RunnerProcess] = None

        async with self.lock:
            if self.shutdown_event.is_set():
                raise RuntimeError(
                    "Server is shutting down. Cannot start new models.")

            # Check maximum concurrent models
            active_count = sum(1 for r in self.active_runners.values()
                               if r.is_alive() and r.status not in ["stopped", "crashed"])

            if (model_alias not in self.active_runners and
                    active_count >= self.config.system.max_concurrent_models):
                raise RuntimeError(
                    f"Maximum concurrent models ({self.config.system.max_concurrent_models}) tercapai."
                )

            if model_alias in self.active_runners:
                runner = self.active_runners[model_alias]

                if not runner.is_alive():
                    logger.warning(
                        f"[{model_alias}] Runner terdeteksi mati.")
                    self._release_port(runner.port)
                    del self.active_runners[model_alias]
                    runner = None  # Set ke None agar memicu Cold Start

                elif runner.status == "loading" or runner.status == "starting":
                    # PENTING: Update last_used_time meski sedang loading
                    # Ini mencegah idle watchdog menghentikan model saat loading
                    runner.last_used_time = time.time()
                    logger.info(
                        f"[{model_alias}] Request diterima saat status '{runner.status}'.")

                else:
                    runner.last_used_time = time.time()
                    return runner

            if runner is None:
                # Get model file size for VRAM estimation
                model_conf = self.config.models[model_alias]
                # Use resolved path
                model_path = Path(model_conf.get_resolved_path())
                model_size_mb = model_path.stat().st_size / (1024**2)

                # Better VRAM estimation formula:
                # GGUF models typically use 1.5-3x file size in VRAM depending on:
                # - n_ctx (KV cache)
                # - n_batch (batch processing memory)
                # - CUDA overhead
                n_ctx = model_conf.params.n_ctx
                vram_multiplier = self.config.system.vram_multiplier

                # Base estimation: file_size * multiplier
                base_vram = model_size_mb * vram_multiplier

                # Additional KV cache estimation (rough approximation)
                # For typical models: ~0.5MB per 1K context
                kv_cache_estimate = (n_ctx / 1024) * 50  # 50MB per 1K context

                # Total estimation with overhead
                estimated_vram_needed_mb = base_vram + \
                    kv_cache_estimate + 150  # 150MB CUDA overhead

                # Minimum threshold from config
                min_vram_required = self.config.system.min_vram_required
                estimated_vram_needed_mb = max(
                    estimated_vram_needed_mb, min_vram_required)

                logger.info(
                    f"[{model_alias}] Preparing to load | File: {model_size_mb:.0f} MB | "
                    f"Estimated VRAM: {estimated_vram_needed_mb:.0f} MB (base: {base_vram:.0f} + KV: {kv_cache_estimate:.0f} + overhead)"
                )

                metrics["models_loaded_total"] += 1  # Track metric
                new_port = self._allocate_port()

                runner = RunnerProcess(
                    alias=model_alias,
                    config=model_conf,
                    port=new_port,
                    llama_server_path=self.config.system.llama_server_path,
                    system_config=self.config.system
                )

                runner.status = "starting"
                self.active_runners[model_alias] = runner

                # Acquire load_lock and track model load start
                # This ensures sequential loading and accurate VRAM measurement
                await self.vram_tracker.track_model_load_start(model_alias, new_port)

                # Use can_load_model() for real-time VRAM check
                can_load, available_mb, vram_message = self.vram_tracker.can_load_model(
                    estimated_vram_mb=estimated_vram_needed_mb,
                    safety_buffer_mb=200  # 200MB safety buffer
                )

                logger.info(f"[{model_alias}] VRAM check: {vram_message}")

                if not can_load:
                    # Not enough VRAM - cleanup and reject
                    loaded_models = [
                        alias for alias, r in self.active_runners.items()
                        if r.is_alive() and r.status == "ready"
                    ]

                    # Release resources
                    self._release_port(new_port)
                    del self.active_runners[model_alias]

                    # Release VRAM tracker lock (this also releases load_lock)
                    await self.vram_tracker.track_model_load_failed(
                        model_alias,
                        f"Insufficient VRAM: need {estimated_vram_needed_mb + 200:.0f} MB, have {available_mb:.0f} MB"
                    )

                    raise InsufficientVRAMError(
                        model_alias=model_alias,
                        required_mb=estimated_vram_needed_mb + 200,
                        available_mb=available_mb,
                        loaded_models=loaded_models
                    )

                logger.info(
                    f"[{model_alias}] VRAM check passed - proceeding with load")

        # Retry logic with proper error handling
        max_retries = runner.max_retries

        for attempt in range(max_retries + 1):
            # Respect shutdown event before each attempt
            if self.shutdown_event.is_set():
                logger.warning(
                    f"[{model_alias}] Aborting start due to shutdown")

                # Jika shutdown maka panggil fungsi track_model_load_failed, tapi msgnya mengarah ke shutdown
                await self.vram_tracker.track_model_load_failed(
                    model_alias, "Server is shutting down"
                )

                async with self.lock:
                    if model_alias in self.active_runners:
                        port = runner.port
                        self._release_port(port)
                        del self.active_runners[model_alias]
                raise RuntimeError("Server shutting down")

            try:
                if runner.status == "starting":
                    await runner.start()
                elif runner.status == "loading":
                    await runner._wait_for_ready(timeout=120)

                runner.last_used_time = time.time()

                if model_alias in self.failed_models:
                    del self.failed_models[model_alias]

                # Track VRAM load complete
                await self.vram_tracker.track_model_load_complete(model_alias)

                # Track to Prometheus metrics
                prom_collector = get_prometheus_collector()
                if prom_collector:
                    load_duration = time.time() - runner.started_time if runner.started_time else 0
                    vram_bytes = 0
                    if model_alias in self.vram_tracker.model_tracks:
                        vram_bytes = self.vram_tracker.model_tracks[
                            model_alias].current_vram_used_mb * 1024 * 1024
                    prom_collector.record_model_load_complete(
                        model_alias, load_duration, vram_bytes)
                    # Register model to initialize gauges (queue_depth, active_requests)
                    prom_collector.register_model(model_alias)
                    # Update loaded models count
                    loaded_count = sum(
                        1 for r in self.active_runners.values() if r.status == "ready")
                    prom_collector.set_models_loaded_count(loaded_count)

                return runner

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"[{model_alias}] Start attempt {attempt + 1}/{max_retries + 1} failed: {error_msg}")

                # Is error retriable?
                if not self._is_retriable_error(error_msg):
                    logger.error(
                        f"[{model_alias}] Permanent error detected. No retry. | Error: {error_msg}"
                    )

                    # Track failed model
                    self.failed_models[model_alias] = {
                        "error": error_msg,
                        "attempts": attempt + 1
                    }

                    # Track VRAM load failed
                    await self.vram_tracker.track_model_load_failed(model_alias, error_msg)

                    # Track to Prometheus metrics
                    prom_collector = get_prometheus_collector()
                    if prom_collector:
                        prom_collector.record_model_load_failed(model_alias)

                    # Cleanup
                    async with self.lock:
                        if model_alias in self.active_runners:
                            port = runner.port
                            self._release_port(port)
                            del self.active_runners[model_alias]

                    raise RuntimeError(
                        f"Model '{model_alias}' failed to start due to configuration error | Error: {error_msg} "
                    )

                # Last attempt failed
                if attempt >= max_retries:
                    logger.error(
                        f"[{model_alias}] All {max_retries + 1} attempts failed. Giving up.")

                    # Track failed model
                    self.failed_models[model_alias] = {
                        "error": error_msg,
                        "attempts": attempt + 1
                    }

                    # Cleanup
                    async with self.lock:
                        if model_alias in self.active_runners:
                            port = runner.port
                            self._release_port(port)
                            del self.active_runners[model_alias]

                    raise RuntimeError(
                        f"Model '{model_alias}' failed after {max_retries + 1} attempts. | Last error: {error_msg}"
                    )

                # Retry logic
                logger.info(
                    f"[{model_alias}] Retry {attempt + 1}/{max_retries}...")

                # CRITICAL FIX: Release load_lock before retry to prevent deadlock
                # The lock will be re-acquired at the start of the next retry attempt
                await self.vram_tracker.track_model_load_failed(
                    model_alias, f"Retry attempt {attempt + 1} - releasing lock"
                )

                # Wait before retry (with shutdown check)
                for _ in range(4):  # 2 seconds total, check every 0.5s
                    if self.shutdown_event.is_set():
                        logger.warning(
                            f"[{model_alias}] Aborting retry due to shutdown")

                        async with self.lock:
                            if model_alias in self.active_runners:
                                port = runner.port
                                self._release_port(port)
                                del self.active_runners[model_alias]
                        raise RuntimeError("Server shutting down")
                    await asyncio.sleep(0.5)

                # Re-acquire load_lock for retry attempt
                await self.vram_tracker.track_model_load_start(model_alias, runner.port)

                # Reset runner for retry
                runner.status = "starting"
                runner.retry_count = attempt + 1  # Properly increment

        # Should never reach here
        raise RuntimeError(
            f"Unexpected error in retry logic for {model_alias}")

    async def eject_model(self, model_alias: str) -> bool:
        async with self.lock:
            if model_alias in self.active_runners:
                runner = self.active_runners[model_alias]
                port = runner.port  # Simpan port sebelum stop
                await runner.stop()
                self._release_port(port)  # Release port
                del self.active_runners[model_alias]
                metrics["models_ejected_total"] += 1  # Track metric

                # Notify VRAM Tracker tentang eject
                await self.vram_tracker.track_model_eject(model_alias)

                # Track to Prometheus metrics
                prom_collector = get_prometheus_collector()
                if prom_collector:
                    prom_collector.record_model_eject(model_alias)
                    # Update loaded models count
                    loaded_count = sum(
                        1 for r in self.active_runners.values() if r.status == "ready")
                    prom_collector.set_models_loaded_count(loaded_count)

                logger.info(
                    f"[{model_alias}] Berhasil di-eject. Port {port} dikembalikan ke pool.")
                return True
            else:
                logger.warning(
                    f"[{model_alias}] - sedang tidak sedang berjalan.")
                return False

    async def stop_all_runners(self):
        logger.info("Mematikan semua runner yang aktif.")

        # Stop VRAM monitoring first
        await self.vram_tracker.stop_monitoring()

        async with self.lock:
            if not self.active_runners:
                logger.info("No active runners to stop.")
                return

            # Log all models yang akan di-stop
            model_list = list(self.active_runners.keys())
            logger.info(f"Stopping {len(model_list)} models: {model_list}")

            # Stop each runner with logging
            for alias, runner in self.active_runners.items():
                logger.info(f"Stopping runner: {alias}")
                await runner.stop()
                logger.info(f"Runner stopped: {alias}")

        # Shutdown VRAM tracker
        self.vram_tracker.shutdown()
        logger.info("Semua runner dimatikan.")

    def _is_retriable_error(self, error_msg: str) -> bool:
        """
        Determine if an error is retriable or permanent.
        Returns False for configuration errors that won't be fixed by retrying.
        """
        # Configuration errors that should not be retried
        non_retriable_patterns = [
            "Unsupported cache type",
            "error while handling argument",
            "Model file not found",
            "Invalid model path",
            "GGML_ASSERT",
            "llama_model_load",
            "unknown argument",
            "invalid argument",
            "failed to load model",
        ]

        error_lower = error_msg.lower()
        for pattern in non_retriable_patterns:
            if pattern.lower() in error_lower:
                return False

        # Other errors (timeouts, connection issues, etc.) are retriable
        return True

    async def get_model_status(self, model_alias: str) -> Dict:
        async with self.lock:
            if model_alias not in self.config.models:
                raise LookupError(f"Model '{model_alias}' tidak dikenal.")

            if model_alias in self.active_runners:
                runner = self.active_runners[model_alias]
                if runner.status == "crashed":
                    return {"status": "crashed", "detail": runner.startup_error}

                return {"status": runner.status, "port": runner.port}
            else:
                return {"status": "stopped"}
