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

        # Check apakah model recently active (ada request dalam 10 menit terakhir).
        self.recent_activity_window = 600  # 10 minutes

    async def preload_models(self):
        """Preload models yang di-specify di config."""
        if not self.config.system.preload_models:
            logger.info("No models configured for preloading")
            return

        logger.info(f"Preloading models: {self.config.system.preload_models}")

        for model_alias in self.config.system.preload_models:
            if self.shutdown_event.is_set():
                logger.info("Shutdown detected. Stopping preload.")
                return

            if model_alias not in self.config.models:
                logger.warning(
                    f"Model '{model_alias}' di preload_models tidak ada di config")
                continue

            try:
                logger.info(f"Preloading model: {model_alias}")

                runner = await asyncio.wait_for(
                    self.manager.get_runner_for_request(model_alias),
                    timeout=120.0
                )

                logger.info(
                    f"Model '{model_alias}' successfully preloaded at {runner.url}")

            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout preloading model '{model_alias}' (exceeded 120s)")
            except asyncio.CancelledError:
                logger.info(f"Preload cancelled for model '{model_alias}'")
                raise
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

    def is_recently_active(self, model_alias: str) -> bool:
        """
        Check apakah model recently active (ada request dalam 10 menit terakhir).
        Ini prevent preload model yang sudah idle timeout.
        """
        if model_alias not in self.last_request_time:
            return False

        time_since_last_request = time.time(
        ) - self.last_request_time[model_alias]
        return time_since_last_request < self.recent_activity_window

    async def maintain_warm_models(self):
        """Background task untuk keep popular models warm."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=300  # Check setiap 5 menit
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                keep_warm_count = self.config.system.keep_warm_models
                if keep_warm_count == 0:
                    continue

                popular_models = self.get_popular_models(top_n=keep_warm_count)

                if not popular_models:
                    continue

                # Filter hanya model yang recently active
                recently_active_models = [
                    model for model in popular_models
                    if self.is_recently_active(model)
                ]

                if not recently_active_models:
                    logger.debug(
                        "No recently active models to keep warm. "
                        f"Popular models {popular_models} are all idle."
                    )
                    continue

                logger.info(
                    f"Maintaining warm models: {recently_active_models}")

                for model_alias in recently_active_models:
                    if self.shutdown_event.is_set():
                        logger.info(
                            "Shutdown detected. Stopping warm maintenance.")
                        return

                    try:
                        async with self.manager.lock:
                            if model_alias in self.manager.active_runners:
                                runner = self.manager.active_runners[model_alias]
                                if runner.is_alive():
                                    # Update last_used_time untuk prevent idle timeout
                                    runner.last_used_time = time.time()
                                    logger.debug(
                                        f"Keeping model '{model_alias}' warm")
                                else:
                                    # Runner died, preload again
                                    logger.info(
                                        f"Re-preloading dead runner: {model_alias}")

                                    # Preload dengan error handling yang lebih baik
                                    try:
                                        await asyncio.wait_for(
                                            self.manager.get_runner_for_request(
                                                model_alias),
                                            timeout=120.0
                                        )
                                        logger.info(
                                            f"Successfully re-preloaded '{model_alias}'")
                                    except asyncio.TimeoutError:
                                        logger.error(
                                            f"Timeout re-preloading '{model_alias}' (120s). "
                                            "Skipping for this cycle."
                                        )
                                        # Don't retry immediately, wait for next cycle
                                        continue
                            else:
                                # Model tidak running, cek apakah worth preloading
                                time_since_last_request = time.time() - self.last_request_time.get(model_alias, 0)

                                if time_since_last_request < self.config.system.idle_timeout_sec:
                                    # Hanya preload jika belum melewati idle timeout
                                    logger.info(
                                        f"Preloading popular model: {model_alias}")

                                    try:
                                        await asyncio.wait_for(
                                            self.manager.get_runner_for_request(
                                                model_alias),
                                            timeout=120.0
                                        )
                                        logger.info(
                                            f"Successfully preloaded '{model_alias}'")
                                    except asyncio.TimeoutError:
                                        logger.error(
                                            f"Timeout preloading '{model_alias}' (120s). "
                                            "Skipping for this cycle."
                                        )
                                        continue
                                else:
                                    logger.debug(
                                        f"Skipping preload for '{model_alias}': "
                                        f"Last request was {time_since_last_request:.0f}s ago "
                                        f"(exceeds idle timeout of {self.config.system.idle_timeout_sec}s)"
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
        try:
            await self.preload_models()
        except asyncio.CancelledError:
            logger.info("Preload cancelled during startup")
            return

        self.warmup_task = asyncio.create_task(self.maintain_warm_models())

        logger.info("Model warmup manager started")

    async def stop(self):
        """Stop warmup manager."""
        logger.info("Stopping warmup manager...")

        if self.warmup_task and not self.warmup_task.done():
            self.warmup_task.cancel()
            try:
                await asyncio.wait_for(self.warmup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("Warmup manager stopped")
