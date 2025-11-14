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
    def __init__(self, alias: str, config: ModelConfig, port: int, llama_server_path: str):
        self.alias = alias
        self.config = config
        self.port = port
        self.llama_server_path = llama_server_path
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
            "--ctx-size", str(params.n_ctx),
            "--batch-size", str(params.n_batch),
            "--no-mmap",
        ]

        if params.embedding:
            command.append("--embedding")
        if params.chat_template:
            command.extend(["--chat-template", params.chat_template])

        logger.info(f"[{self.alias}] di Port {self.port}.")
        logger.debug(f"[{self.alias}] Command: {' '.join(command)}")
        logger.info(f"[{self.alias}] Log file: {self.log_file}")
        self.startup_error = None

        # Buka log file untuk stdout dan stderr
        log_handle = open(self.log_file, 'w')

        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_handle,
            stderr=subprocess.STDOUT  # Redirect stderr ke stdout (ke log file)
        )

        # Simpan handle untuk di-close nanti
        self.log_handle = log_handle

        await self._wait_for_ready()

        self.last_used_time = time.time()
        self.status = "ready"
        logger.info(f"[{self.alias}] SIAP di {self.url}")

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
                        logger.info(
                            f"[{self.alias}] | Tunggu.")
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
        self.port_pool = set(range(8085, 8585))  # Pool 500 ports
        self.used_ports = set()
        self.lock = asyncio.Lock()
        self.gpu_devices = config.system.gpu_devices
        self.gpu_allocator_index = 0
        self.gpu_loads: Dict[int, float] = {
            gpu: 0.0 for gpu in self.gpu_devices}
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
                                gpu_id = runner.gpu_id
                                await runner.stop()
                                self._release_port(port)
                                self._update_gpu_load(gpu_id)
                                runners_to_stop.append(alias)
                            continue

                        # Check idle timeout untuk model yang ready
                        if (current_time - runner.last_used_time) > timeout:
                            logger.info(
                                f"Model '{alias}' melebihi waktu {timeout}d.")
                            port = runner.port
                            gpu_id = runner.gpu_id
                            await runner.stop()
                            self._release_port(port)
                            self._update_gpu_load(gpu_id)
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

        # Menyimpan runner
        runner: Optional[RunnerProcess] = None

        async with self.lock:
            # Check maximum concurrent models
            active_count = sum(1 for r in self.active_runners.values()
                               if r.is_alive() and r.status not in ["stopped", "crashed"])

            if (model_alias not in self.active_runners and
                    active_count >= self.config.system.max_concurrent_models):
                raise RuntimeError(
                    f"Maximum concurrent models ({self.config.system.max_concurrent_models}) "
                    f"tercapai. Eject model lain terlebih dahulu atau tunggu idle timeout."
                )

            if model_alias in self.active_runners:
                runner = self.active_runners[model_alias]

                if not runner.is_alive():
                    logger.warning(
                        f"[{model_alias}] Runner terdeteksi mati.")
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
                    llama_server_path=self.config.system.llama_server_path
                )

                runner.status = "starting"
                self.active_runners[model_alias] = runner

        try:
            if runner.status == "starting":
                await runner.start()
            elif runner.status == "loading":
                await runner._wait_for_ready(timeout=120)
            runner.last_used_time = time.time()

            return runner

        except Exception as e:
            logger.error(f"Gagal total men-start {model_alias}: {e}")

            if runner.retry_count < runner.max_retries:
                runner.retry_count += 1
                logger.info(
                    f"[{model_alias}] Retry {runner.retry_count}/{runner.max_retries}."
                )
                await asyncio.sleep(2)  # Wait sebentar sebelum retry

                # Reset status dan coba lagi
                runner.status = "starting"
                return await self.get_runner_for_request(model_alias)
            else:
                # Give up setelah max retries
                logger.error(
                    f"[{model_alias}] Gagal setelah {runner.max_retries} retry. Giving up."
                )

                async with self.lock:
                    if model_alias in self.active_runners:
                        port = runner.port
                        self._release_port(port)
                        del self.active_runners[model_alias]
                raise e

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
