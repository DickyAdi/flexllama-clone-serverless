import time
import httpx
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .metrics import metrics
from .config import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class RunnerProcess:
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
        self.max_retries = 2  # Max 2 kali

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
        command = [
            self.llama_server_path, "--model", self.config.model_path,
            "--host", "127.0.0.1", "--port", str(self.port),
            "--n-gpu-layers", str(params.n_gpu_layers),
            "--ctx-size", str(params.n_ctx), "--mlock",
            "--rope-freq-base", str(params.rope_freq_base), "-nkvo"
        ]

        # Batch size (with per-model override)
        batch_size = params.batch_override if params.batch_override else params.n_batch
        command.extend(["--batch-size", str(batch_size)])

        # Parallel requests (with per-model override)
        parallel = params.parallel_override if params.parallel_override else self.system_config.parallel_requests
        command.extend(["--parallel", str(parallel)])

        # CPU threads
        command.extend(["--threads", str(self.system_config.cpu_threads)])

        # Flash Attention
        command.extend(["-fa", self.system_config.flash_attention])

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

        logger.info(f"[{self.alias}] di Port {self.port}.")
        logger.debug(f"[{self.alias}] Command: {' '.join(command)}")
        logger.info(f"[{self.alias}] Log file: {self.log_file}")
        logger.info(
            f"[{self.alias}] Performance: threads={self.system_config.cpu_threads}, "
            f"flash_attn={self.system_config.flash_attention}, mmap={self.system_config.use_mmap}"
        )
        self.startup_error = None

        # Buka log file untuk stdout dan stderr
        log_handle = open(self.log_file, 'w')

        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_handle,
            stderr=subprocess.STDOUT
        )

        # Simpan handle untuk di-close nanti
        self.log_handle = log_handle

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
            f"[{self.alias}] READY at {self.url} | "
            f"Total: {total_startup_time:.2f}s "
            f"(subprocess: {subprocess_time:.2f}s, loading: {health_check_time:.2f}s)"
        )

    async def stop(self):
        if not self.is_alive() or self.process is None:
            self.status = "stopped"
            return
        logger.info(
            f"[{self.alias}] Menghentikan proses (Port {self.port}).")
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=10.0)
            logger.info(f"[{self.alias}] Berhasil dihentikan.")
        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.alias}] Gagal terminate.")
            self.process.kill()
            await self.process.wait()

        self.process = None
        self.status = "stopped"

        # Close log file handle
        if hasattr(self, 'log_handle'):
            self.log_handle.close()

    async def _wait_for_ready(self, timeout=120):
        self.status = "loading"
        start_time = time.time()
        last_log_time = 0

        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                if not self.is_alive():
                    try:
                        with open(self.log_file, 'r') as f:
                            lines = f.readlines()
                            self.startup_error = ''.join(lines)
                    except Exception as e:
                        self.startup_error = f"Proses crash, gagal membaca log: {e}"

                    self.status = "crashed"
                    logger.error(
                        f"[{self.alias}] | Crash. Error: {self.startup_error}")
                    raise Exception(
                        f"Gagal memulai model. Error: {self.startup_error}")

                try:
                    response = await client.get(f"{self.url}/health", timeout=1.0)
                    if response.status_code == 200:
                        self.status = "ready"
                        return
                    elif response.status_code == 503:
                        current_time = time.time()

                        if current_time - last_log_time >= 5.0:
                            elapsed = current_time - start_time
                            logger.info(
                                f"[{self.alias}] Loading... ({elapsed:.1f}s elapsed)")
                            last_log_time = current_time

                        self.status = "loading"
                        await asyncio.sleep(1.0)
                    else:
                        logger.warning(
                            f"[{self.alias}] | Status {response.status_code}.")
                        await asyncio.sleep(1.0)
                except httpx.ConnectError:
                    logger.info(
                        f"[{self.alias}] | Gagal terhubung - menunggu.")
                    self.status = "starting"
                    await asyncio.sleep(0.5)

            await self.stop()
            self.status = "crashed"
            raise TimeoutError(
                f"Runner {self.alias} gagal start dalam {timeout} detik (timeout)."
                f"Check log di {self.log_file}"
            )


class ModelManager:
    def __init__(self, config: AppConfig, shutdown_event):
        self.config = config
        self.shutdown_event = shutdown_event
        self.active_runners: Dict[str, RunnerProcess] = {}
        self.port_pool = set(range(8085, 8585))
        self.used_ports = set()
        self.lock = asyncio.Lock()
        self.gpu_devices = config.system.gpu_devices
        self.gpu_allocator_index = 0
        self.gpu_loads: Dict[int, float] = {
            gpu: 0.0 for gpu in self.gpu_devices}

        # {model_alias: {error: str, attempts: int}}
        self.failed_models: Dict[str, Dict] = {}

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
                    active_aliases = list(self.active_runners.keys())

                for alias in active_aliases:
                    # Check shutdown sebelum process setiap runner
                    if self.shutdown_event.is_set():
                        return

                    async with self.lock:
                        runner = self.active_runners.get(alias)
                        if runner is None:
                            continue

                        # Jika stuck di loading terlalu lama, stop
                        if runner.status in ["loading", "starting"]:
                            if runner.started_time and (current_time - runner.started_time) > max_time:
                                logger.warning(
                                    f"Model '{alias}' stuck di status '{runner.status}' "
                                    f"lebih dari {max_time}s. Forcing stop."
                                )
                                port = runner.port
                                await runner.stop()
                                self._release_port(port)
                                runners_to_stop.append(alias)
                            continue

                        # Check idle timeout untuk model yang ready
                        if (current_time - runner.last_used_time) > timeout:
                            logger.info(
                                f"Model '{alias}' melebihi waktu {timeout}d.")
                            port = runner.port
                            await runner.stop()
                            self._release_port(port)
                            runners_to_stop.append(alias)

                if runners_to_stop:
                    async with self.lock:
                        for alias in runners_to_stop:
                            if alias in self.active_runners:
                                del self.active_runners[alias]
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
                    f"Model '{model_alias}' has failed {failed_info['attempts']} times. "
                    f"Last error: {failed_info['error']}... "
                    f"Fix configuration before retrying."
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
                    logger.info(
                        f"[{model_alias}] Request diterima saat status '{runner.status}'.")

                else:
                    runner.last_used_time = time.time()
                    return runner

            if runner is None:
                logger.info(f"[{model_alias}] sedang bersiap diri.")
                metrics["models_loaded_total"] += 1  # Track metric
                model_conf = self.config.models[model_alias]
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

        # Retry logic with proper error handling
        max_retries = runner.max_retries

        for attempt in range(max_retries + 1):
            # Respect shutdown event before each attempt
            if self.shutdown_event.is_set():
                logger.warning(
                    f"[{model_alias}] Aborting start due to shutdown")
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

                return runner

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"[{model_alias}] Start attempt {attempt + 1}/{max_retries + 1} failed: {error_msg}")
                logger.error(f"Gagal total men-start {model_alias}: {e}")

                # Is error retriable?
                if not self._is_retriable_error(error_msg):
                    logger.error(
                        f"[{model_alias}] Permanent error detected. No retry. "
                        f"Error: {error_msg}"
                    )

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
                        f"Model '{model_alias}' failed to start due to configuration error. "
                        f"Error: {error_msg} "
                        f"Please check config and llama-server arguments."
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
                        f"Model '{model_alias}' failed after {max_retries + 1} attempts. "
                        f"Last error: {error_msg}"
                    )

                # Retry logic
                logger.info(
                    f"[{model_alias}] Retry {attempt + 1}/{max_retries}...")

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

                logger.info(
                    f"[{model_alias}] Berhasil di-eject. Port {port} dikembalikan ke pool.")
                return True
            else:
                logger.warning(
                    f"[{model_alias}] - sedang tidak sedang berjalan.")
                return False

    async def stop_all_runners(self):
        logger.info("Mematikan semua runner yang aktif.")
        async with self.lock:
            for runner in self.active_runners.values():
                await runner.stop()
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
