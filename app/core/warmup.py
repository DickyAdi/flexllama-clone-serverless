import time
import asyncio
import logging
from typing import Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelWarmupManager:
    """Manage model pre-loading dan keep-warm strategy."""

    def __init__(self, manager, config, shutdown_event):
        self.manager = manager
        self.config = config
        self.shutdown_event = shutdown_event

        # Track usage statistics
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_request_time: Dict[str, float] = {}

        # Track loading
        self.preload_task = None
        self.warmup_task = None

    async def preload_models(self):
        """Preload models yang di-specify di config."""
        if not self.config.system.preload_models:
            logger.info("No models configured for preloading")
            return

        logger.info(f"Preloading models: {self.config.system.preload_models}")

        for model_alias in self.config.system.preload_models:
            # Check shutdown event sebelum preload
            if self.shutdown_event.is_set():
                logger.info("Shutdown detected. Stopping preload.")
                return

            if model_alias not in self.config.models:
                logger.warning(
                    f"Model '{model_alias}' di preload_models tidak ada di config")
                continue

            try:
                logger.info(f"Preloading model: {model_alias}")

                # Wrap dengan timeout untuk prevent hang
                runner = await asyncio.wait_for(
                    self.manager.get_runner_for_request(model_alias),
                    timeout=120.0  # 2 menit max per model
                )

                logger.info(
                    f"Model '{model_alias}' successfully preloaded at {runner.url}")

            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout preloading model '{model_alias}' (exceeded 120s)")
            except asyncio.CancelledError:
                logger.info(f"Preload cancelled for model '{model_alias}'")
                raise  # Re-raise untuk proper cancellation
            except Exception as e:
                logger.error(f"Failed to preload model '{model_alias}': {e}")

    def record_request(self, model_alias: str):
        """Record bahwa model di-request (untuk popularity tracking)."""
        self.request_counts[model_alias] += 1
        self.last_request_time[model_alias] = time.time()

    def get_popular_models(self, top_n: int = 5) -> list[str]:
        """Get top N most popular models berdasarkan request count."""
        sorted_models = sorted(
            self.request_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [model for model, count in sorted_models[:top_n]]

    async def maintain_warm_models(self):
        """Background task untuk keep popular models warm."""
        try:
            while not self.shutdown_event.is_set():
                # Use wait_for dengan shutdown_event sebagai alternative
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=300  # Check setiap 5 menit
                    )
                    # Jika sampai sini, berarti shutdown_event di-set
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, lanjut maintain warm models
                    pass

                keep_warm_count = self.config.system.keep_warm_models
                if keep_warm_count == 0:
                    continue

                popular_models = self.get_popular_models(top_n=keep_warm_count)

                if not popular_models:
                    continue

                logger.info(f"Maintaining warm models: {popular_models}")

                for model_alias in popular_models:
                    # Check shutdown sebelum process setiap model
                    if self.shutdown_event.is_set():
                        logger.info(
                            "Shutdown detected. Stopping warm maintenance.")
                        return

                    try:
                        # Check jika model sudah running
                        async with self.manager.lock:
                            if model_alias in self.manager.active_runners:
                                runner = self.manager.active_runners[model_alias]
                                if runner.is_alive():
                                    # Update last_used_time agar tidak di-eject
                                    runner.last_used_time = time.time()
                                    logger.debug(
                                        f"Keeping model '{model_alias}' warm")
                                else:
                                    # Runner died, preload again dengan timeout
                                    logger.info(
                                        f"Re-preloading dead runner: {model_alias}")
                                    await asyncio.wait_for(
                                        self.manager.get_runner_for_request(
                                            model_alias),
                                        timeout=120.0
                                    )
                            else:
                                # Model tidak running, preload dengan timeout
                                logger.info(
                                    f"Preloading popular model: {model_alias}")
                                await asyncio.wait_for(
                                    self.manager.get_runner_for_request(
                                        model_alias),
                                    timeout=120.0
                                )

                    except asyncio.TimeoutError:
                        logger.error(
                            f"Timeout maintaining warm model '{model_alias}'")
                    except Exception as e:
                        logger.error(
                            f"Failed to maintain warm model '{model_alias}': {e}")

        except asyncio.CancelledError:
            logger.info("Warm maintenance task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in maintain_warm_models: {e}")

    async def start(self):
        """Start warmup manager tasks."""
        # Preload configured models
        try:
            await self.preload_models()
        except asyncio.CancelledError:
            logger.info("Preload cancelled during startup")
            return

        # Start background task untuk maintain warm models
        self.warmup_task = asyncio.create_task(self.maintain_warm_models())

        logger.info("Model warmup manager started")

    async def stop(self):
        """Stop warmup manager."""
        logger.info("Stopping warmup manager.")

        if self.warmup_task and not self.warmup_task.done():
            self.warmup_task.cancel()
            try:
                await asyncio.wait_for(self.warmup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("Warmup manager stopped")
