"""
Model Warmup and Pre-loading Module

Modul ini menyediakan sistem pre-loading model saat startup dan keep-warm
strategy untuk menjaga model populer tetap loaded di VRAM.

Components:
    - ModelWarmupManager: Manager untuk preloading dan warmup strategy

Features:
    - Preload models saat startup (configurable via preload_models)
    - Keep-warm strategy untuk model populer berdasarkan usage statistics
    - VRAM-aware loading (skip models jika VRAM tidak cukup)
    - Popularity tracking berdasarkan request count
    - Recent activity tracking untuk prevent idle preload

Preload Modes:
    - ["*"]: Load semua model yang ada di config
    - ["model1", "model2"]: Load model spesifik
    - []: Tidak preload apapun

Warmup Strategy:
    - Setiap 5 menit, check top N model populer (keep_warm_models)
    - Jika model tidak running tapi recently active, preload
    - Jika model running, update last_used_time untuk prevent idle timeout
    - Skip models yang gagal load karena VRAM constraint

Usage:
    warmup_manager = ModelWarmupManager(manager, config, shutdown_event)
    await warmup_manager.start()

    # Record request untuk popularity tracking
    warmup_manager.record_request("qwen3-8b")

    # Check if model is warm
    is_warm = warmup_manager.is_model_warm("qwen3-8b")

    # Stop
    await warmup_manager.stop()

Configuration (via system config):
    - preload_models: List models to preload
    - preload_delay_sec: Delay between model loads
    - keep_warm_models: Number of popular models to keep warm
    - timeout_warmup_sec: Timeout for model loading
    - wait_ready_sec: Max wait for model ready status
"""

import time
import asyncio
import logging
from typing import Dict, List, Set
from collections import defaultdict

from app.core import config

logger = logging.getLogger(__name__)


class ModelWarmupManager:
    """
    Manage model pre-loading dan keep-warm strategy.

    Handles:
    - Preload models saat startup
    - Keep popular models warm (prevent idle timeout)
    - Track model popularity berdasarkan request count
    """

    def __init__(self, manager, config, shutdown_event):
        self.manager = manager
        self.config = config
        self.shutdown_event = shutdown_event

        # Track usage statistics
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_request_time: Dict[str, float] = {}

        # Track models that failed to load due to VRAM
        # These should not be retried in warm maintenance
        self.vram_failed_models: Set[str] = set()

        # Track loading
        self.preload_task = None
        self.warmup_task = None

        # Check apakah model recently active.
        self.recent_activity_window = 600

    def _resolve_preload_models(self) -> List[str]:
        """
        Resolve preload_models configuration ke list of model aliases.

        Supports:
        - ["*"] : Load semua model yang ada di config
        - ["model1", "model2", ...] : Load model spesifik
        - [] : Tidak preload apapun

        Returns:
            List of model aliases to preload
        """
        preload_config = self.config.system.preload_models

        if not preload_config:
            return []

        # Check for wildcard ["*"]
        if len(preload_config) == 1 and preload_config[0] == "*":
            all_models = list(self.config.models.keys())
            logger.info(
                f"[Preload] Wildcard '*' detected. Will preload ALL {len(all_models)} models: {all_models}"
            )
            return all_models

        # Validate specific model aliases
        valid_models = []
        for model_alias in preload_config:
            if model_alias == "*":
                logger.warning(
                    "[Preload] Wildcard '*' should be used alone, not mixed with other aliases. Ignoring '*' and using specific aliases."
                )
                continue

            if model_alias in self.config.models:
                valid_models.append(model_alias)
            else:
                logger.warning(
                    f"[Preload] Model '{model_alias}' tidak ada di config. Skipping."
                )

        return valid_models

    async def _load_single_model(self, model_alias: str, index: int, total: int) -> bool:
        """
        Load a single model with detailed logging and error handling.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(
                f"[Preload] [{index}/{total}] Loading model: {model_alias}")

            start_time = time.time()

            # Get runner
            runner = await asyncio.wait_for(
                self.manager.get_runner_for_request(model_alias),
                timeout=self.config.system.timeout_warmup_sec
            )

            # Wait for ready status
            max_wait_ready = self.config.system.wait_ready_sec
            wait_start = time.time()

            while runner.status not in ["ready", "crashed", "stopped"]:
                if time.time() - wait_start > max_wait_ready:
                    logger.warning(
                        f"[Preload] [{index}/{total}] Model '{model_alias}' stuck at '{runner.status}' "
                        f"after {max_wait_ready}s"
                    )
                    return False

                if self.shutdown_event.is_set():
                    logger.info(
                        "[Preload] Shutdown detected while waiting for ready.")
                    return False

                await asyncio.sleep(1)

            if runner.status == "ready":
                load_time = time.time() - start_time

                # Get VRAM info
                vram_after = self.manager.vram_tracker.get_vram_report()

                logger.info(
                    f"[Preload] [{index}/{total}] SUCCESS | '{model_alias}' loaded at {runner.url} "
                    f"(took {load_time:.1f}s) | VRAM: {vram_after['gpu_info']['used_gb']:.2f} GB used, "
                    f"{vram_after['gpu_info']['free_gb']:.2f} GB free"
                )
                return True
            else:
                logger.error(
                    f"[Preload] [{index}/{total}] FAILED | Model '{model_alias}' ended with status '{runner.status}'"
                )
                return False

        except asyncio.TimeoutError:
            logger.error(
                f"[Preload] [{index}/{total}] TIMEOUT | Model '{model_alias}' exceeded {config.system.timeout_warmup_sec}s during loading")
            return False
        except Exception as e:
            logger.error(
                f"[Preload] [{index}/{total}] ERROR | Model '{model_alias}': {e}")
            return False

    async def preload_models_queued(self, max_parallel: int = 2):
        """
        Preload models with queued loading.

        Note: Due to load_lock in VRAMTracker, models load sequentially even when
        multiple tasks are created. This prevents VRAM race conditions but means
        max_parallel is effectively a batch size for logging purposes.

        VRAM checking is handled by manager.get_runner_for_request, not here.
        """
        models_to_preload = self._resolve_preload_models()

        if not models_to_preload:
            logger.info("[Preload] No models configured for preloading")
            return

        total_models = len(models_to_preload)
        max_concurrent = self.config.system.max_concurrent_models
        preload_delay = self.config.system.preload_delay_sec

        logger.info(
            f"[Preload] Starting preload QUEUED for {total_models} models: {models_to_preload}"
        )

        logger.info(
            f"[Preload] System max_concurrent_models={max_concurrent} | Batch size={max_parallel} | Delay between batches={preload_delay}s"
        )

        successful_loads = 0
        failed_loads = 0
        skipped_loads = 0

        for batch_start in range(0, total_models, max_parallel):
            if self.shutdown_event.is_set():
                logger.info("[Preload] Shutdown detected. Stopping preload.")
                return

            batch_end = min(batch_start + max_parallel, total_models)
            batch = models_to_preload[batch_start:batch_end]

            logger.info(
                f"[Preload] Batch {batch_start//max_parallel + 1}: Queuing {len(batch)} models: {batch}"
            )

            # VRAM check is handled by manager.get_runner_for_request
            # No duplicate check needed here - manager will reject if VRAM insufficient

            # Create tasks for batch (they will be processed sequentially due to load_lock)
            load_tasks = []
            for model_alias in batch:
                task = asyncio.create_task(self._load_single_model(
                    model_alias, batch_start + batch.index(model_alias) + 1, total_models))
                load_tasks.append((model_alias, task))

            # Wait for all loads in batch to complete
            batch_results = await asyncio.gather(*[task for _, task in load_tasks], return_exceptions=True)

            # Process results
            for (model_alias, _), result in zip(load_tasks, batch_results):
                if isinstance(result, Exception):
                    failed_loads += 1
                    logger.error(
                        f"[Preload] Failed to load {model_alias}: {result}")

                    # Check if VRAM related
                    error_str = str(result).lower()
                    if "vram" in error_str or "insufficient" in error_str:
                        self.vram_failed_models.add(model_alias)
                        skipped_loads += 1
                        failed_loads -= 1  # Don't double count
                elif result:
                    successful_loads += 1
                else:
                    failed_loads += 1

            # Delay before next batch (except last batch)
            if batch_end < total_models and preload_delay > 0:
                logger.info(
                    f"[Preload] Waiting {preload_delay}s before next batch...")

                for _ in range(preload_delay):
                    if self.shutdown_event.is_set():
                        logger.info(
                            "[Preload] Shutdown detected during delay.")
                        return
                    await asyncio.sleep(1)

        # Summary
        logger.info(
            f"[Preload] COMPLETE | Total: {total_models} | Success: {successful_loads} | Failed: {failed_loads} | VRAM-skipped: {skipped_loads}"
        )

        # Log final VRAM report
        final_vram = self.manager.vram_tracker.get_vram_report()
        logger.info(
            f"[Preload] VRAM STATUS | Used: {final_vram['gpu_info']['used_gb']:.2f} GB | Free: {final_vram['gpu_info']['free_gb']:.2f} GB | Status: {final_vram['status']}"
        )

    async def preload_models_serial(self):
        models_to_preload = self._resolve_preload_models()

        if not models_to_preload:
            logger.info("[Preload] No models configured for preloading")
            return

        total_models = len(models_to_preload)
        max_concurrent = self.config.system.max_concurrent_models
        preload_delay = self.config.system.preload_delay_sec

        logger.info(
            f"[Preload] Starting preload SERIAL for {total_models} models: {models_to_preload}"
        )

        logger.info(
            f"[Preload] System max_concurrent_models={max_concurrent} | Preload delay between batches={preload_delay}s"
        )

        # Warning jika mencoba preload lebih dari max_concurrent_models
        if total_models > max_concurrent:
            logger.warning(
                f"[Preload] Preload count ({total_models}) > max_concurrent_models ({max_concurrent}). "
            )

        successful_loads = 0
        failed_loads = 0
        skipped_loads = 0

        for idx, model_alias in enumerate(models_to_preload, 1):
            if self.shutdown_event.is_set():
                logger.info("[Preload] Shutdown detected. Stopping preload.")
                return

            # VRAM check is handled by manager.get_runner_for_request
            # No duplicate check needed here - manager will reject if VRAM insufficient
            # Log progress
            logger.info(
                f"[Preload] [{idx}/{total_models}] Loading: {model_alias}"
            )

            # Load model
            result = await self._load_single_model(model_alias, idx, total_models)

            if result:
                successful_loads += 1
            else:
                failed_loads += 1

            # Delay before next model (except last)
            if idx < total_models and preload_delay > 0:
                logger.info(
                    f"[Preload] Waiting {preload_delay}s before next model...")

                for _ in range(preload_delay):
                    if self.shutdown_event.is_set():
                        logger.info(
                            "[Preload] Shutdown detected during delay.")
                        return
                    await asyncio.sleep(1)

        # Summary
        logger.info(
            f"[Preload] YEAYYYY | Total: {total_models} | Success: {successful_loads} | Failed: {failed_loads} | Skipped: {skipped_loads}"
        )

        # Log final VRAM report
        final_vram = self.manager.vram_tracker.get_vram_report()
        logger.info(
            f"[Preload] VRAM STATUS | Used: {final_vram['gpu_info']['used_gb']:.2f} GB | Free: {final_vram['gpu_info']['free_gb']:.2f} GB | Status: {final_vram['status']}"
        )

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

                # Filter hanya model yang recently active DAN tidak gagal karena VRAM
                recently_active_models = [
                    model for model in popular_models
                    if self.is_recently_active(model) and model not in self.vram_failed_models
                ]

                if not recently_active_models:
                    logger.debug(
                        f"No recently active models to keep warm. Popular models {popular_models} are all idle or VRAM-failed."
                    )
                    continue

                # Log juga model yang di-skip karena VRAM
                vram_skipped = [
                    m for m in popular_models if m in self.vram_failed_models]
                if vram_skipped:
                    logger.debug(
                        f"Skipping VRAM-failed models: {vram_skipped}")

                logger.info(
                    f"Maintaining warm models: {recently_active_models}")

                for model_alias in recently_active_models:
                    if self.shutdown_event.is_set():
                        logger.info(
                            "Shutdown detected. Stopping warm maintenance.")
                        return

                    try:
                        # CRITICAL FIX: Check state inside lock, but perform loading OUTSIDE
                        # to prevent deadlock (get_runner_for_request also acquires manager.lock)
                        need_reload = False
                        need_preload = False

                        async with self.manager.lock:
                            if model_alias in self.manager.active_runners:
                                runner = self.manager.active_runners[model_alias]
                                if runner.is_alive():
                                    # Update last_used_time untuk prevent idle timeout
                                    runner.last_used_time = time.time()
                                    logger.debug(
                                        f"Keeping model '{model_alias}' warm")
                                else:
                                    # Runner died, mark for reload
                                    need_reload = True
                                    logger.info(
                                        f"Dead runner detected: {model_alias}, will re-preload")
                            else:
                                # Model tidak running, cek apakah worth preloading
                                time_since_last_request = time.time() - self.last_request_time.get(model_alias, 0)

                                if time_since_last_request < self.config.system.idle_timeout_sec:
                                    need_preload = True
                                else:
                                    logger.debug(
                                        f"Skipping preload for '{model_alias}' | Last request was {time_since_last_request:.0f}s ago | (exceeds idle timeout of {self.config.system.idle_timeout_sec}s)"
                                    )

                        # Perform loading OUTSIDE of lock to prevent deadlock
                        if need_reload:
                            logger.info(
                                f"Re-preloading dead runner: {model_alias}")
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
                                    f"Timeout re-preloading '{model_alias}'. Skipping for this cycle."
                                )
                                continue

                        if need_preload:
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
                                    f"Timeout preloading '{model_alias}'. Skipping for this cycle."
                                )
                                continue

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

    async def preload_models(self):
        # Determine if we can use parallel loading
        max_concurrent = self.config.system.max_concurrent_models

        # Use parallel loading if max_concurrent > 1 and mmap is enabled
        use_parallel = max_concurrent > 1 and self.config.system.use_mmap

        if use_parallel:
            # Use queued loading (sequential due to load_lock, but tasks created in batches)
            batch_size = min(2, max_concurrent)
            logger.info(
                f"[Preload] Using QUEUED loading (batch size {batch_size})")
            await self.preload_models_queued(max_parallel=batch_size)
        else:
            # Fallback to serial loading
            logger.info("[Preload] Using SERIAL loading")
            await self.preload_models_serial()

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

    def record_request(self, model_alias: str):
        """Record bahwa model di-request (untuk popularity tracking)."""
        self.request_counts[model_alias] += 1
        self.last_request_time[model_alias] = time.time()

    def clear_vram_failed(self, model_alias: str = None):
        """
        Clear VRAM failed status for a model (or all models).
        Call this when VRAM becomes available (e.g., after model eject).
        """
        if model_alias:
            self.vram_failed_models.discard(model_alias)
            logger.info(
                f"[Warmup] Cleared VRAM-failed status for '{model_alias}'")
        else:
            self.vram_failed_models.clear()
            logger.info("[Warmup] Cleared all VRAM-failed statuses")

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

    def is_model_warm(self, model_alias: str) -> bool:
        """
        Check apakah model termasuk dalam warm models yang harus tetap loaded.

        Returns:
            True jika model adalah warm model (top N popular), False otherwise
        """
        keep_warm_count = self.config.system.keep_warm_models
        if keep_warm_count <= 0:
            return False

        popular_models = self.get_popular_models(top_n=keep_warm_count)
        return model_alias in popular_models
