"""
GPU VRAM Tracking Module

Modul ini menyediakan sistem tracking VRAM GPU yang akurat untuk mencegah
Out of Memory (OOM) errors saat loading multiple models.

Components:
    - ModelVRAMSnapshot: Snapshot VRAM usage pada satu waktu
    - ModelVRAMTracking: Tracking lengkap VRAM untuk satu model
    - VRAMTracker: Main tracker dengan NVML integration

Features:
    - Real-time VRAM monitoring via NVIDIA NVML
    - Per-model VRAM usage tracking (before/after load)
    - Sequential loading dengan load_lock untuk prevent race conditions
    - Automatic VRAM estimation untuk reject model jika tidak cukup
    - Background monitoring dengan periodic snapshots
    - Baseline tracking untuk system overhead

Tracking Flow:
    1. track_model_load_start(): Acquire load_lock, record VRAM before load
    2. Model loading process (llama-server start)
    3. track_model_load_complete(): Record VRAM after load, calculate delta, release lock
    4. track_model_eject(): Remove tracking saat model di-unload

VRAM Status:
    - healthy: Free VRAM >= 2x min_required
    - warning: Free VRAM >= min_required
    - critical: Free VRAM < min_required

Usage:
    vram_tracker = VRAMTracker(gpu_device_index=0, min_vram_required=500)
    vram_tracker.start_monitoring()
    
    # Check sebelum load model
    can_load, available_mb, message = vram_tracker.can_load_model(
        estimated_vram_mb=4000,
        safety_buffer_mb=200
    )
    
    if can_load:
        await vram_tracker.track_model_load_start("qwen3-8b", port=8085)
        # ... load model ...
        await vram_tracker.track_model_load_complete("qwen3-8b")
    
    # Get report
    report = vram_tracker.get_vram_report()

Note:
    Membutuhkan pynvml (NVIDIA Management Library) yang terinstall.
    Hanya support NVIDIA GPUs.
"""

import pynvml
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelVRAMSnapshot:
    """Snapshot VRAM usage untuk satu model pada satu waktu."""
    model_alias: str
    timestamp: datetime
    vram_used_mb: float
    total_vram_used_mb: float
    port: int
    status: str


@dataclass
class ModelVRAMTracking:
    """Tracking VRAM usage untuk satu model."""
    model_alias: str
    port: int
    vram_before_load_used_mb: float
    vram_after_load_used_mb: float = 0.0
    current_vram_used_mb: float = 0.0
    load_start_time: Optional[datetime] = None
    load_end_time: Optional[datetime] = None
    status: str = "loading"  # loading, loaded, failed, ejected

    # History snapshots (keep last 10)
    snapshots: List[ModelVRAMSnapshot] = field(default_factory=list)
    max_snapshots: int = 10

    def add_snapshot(self, snapshot: ModelVRAMSnapshot):
        """Add snapshot to history."""
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def get_average_usage_mb(self) -> float:
        """Get average VRAM usage dari snapshots."""
        if not self.snapshots:
            return self.current_vram_used_mb

        usages = [s.vram_used_mb for s in self.snapshots]
        return sum(usages) / len(usages)

    def get_load_duration_sec(self) -> Optional[float]:
        """Get duration untuk load model (seconds)."""
        if self.load_start_time and self.load_end_time:
            return (self.load_end_time - self.load_start_time).total_seconds()
        return None


class VRAMTracker:
    """
    Track VRAM usage per model secara dinamis dengan metode yang lebih akurat.
    """

    def __init__(self, gpu_device_index: int = 0, min_vram_required: int = 500):
        self.gpu_device_index = gpu_device_index
        self.gpu_handle = None
        self.min_vram_required = min_vram_required  # Configurable minimum VRAM

        # Track models: {model_alias: ModelVRAMTracking}
        self.model_tracks: Dict[str, ModelVRAMTracking] = {}
        self.lock = asyncio.Lock()

        # Lock untuk memastikan hanya satu model yang load pada satu waktu
        self.load_lock = asyncio.Lock()
        self.currently_loading: Optional[str] = None

        # Background monitoring
        self.monitoring = False
        self.monitor_task = None
        self.monitor_interval_sec = 5

        # Baseline VRAM (system overhead)
        self.baseline_vram_used_mb = 0.0

        # Initial free VRAM
        self.initial_free_vram_mb = 0.0

        # Initialize NVML
        self._init_nvml()

    def _init_nvml(self):
        """Initialize NVML and get GPU handle."""
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_device_index)
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)

            # Catat baseline VRAM (system overhead sebelum load model apapun)
            vram_info = self.get_current_vram_info()
            self.baseline_vram_used_mb = vram_info["used_mb"]
            self.initial_free_vram_mb = vram_info["free_mb"]

            logger.info(
                f"[VRAM Tracker] Initialized for GPU {self.gpu_device_index}: {gpu_name} | Baseline VRAM used: {self.baseline_vram_used_mb:.0f} MB | Initial free: {self.initial_free_vram_mb:.0f} MB"
            )
        except Exception as e:
            logger.error(f"[VRAM Tracker] Failed to initialize NVML: {e}")
            raise

    def get_current_vram_info(self) -> Dict[str, float]:
        """Get current VRAM info (total, used, free) in MB."""
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                "total_mb": mem_info.total / (1024 ** 2),
                "used_mb": mem_info.used / (1024 ** 2),
                "free_mb": mem_info.free / (1024 ** 2)
            }
        except Exception as e:
            logger.error(f"[VRAM Tracker] Failed to get VRAM info: {e}")
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

    async def track_model_load_start(self, model_alias: str, port: int):
        """Start tracking a model load. Acquires load_lock for sequential loading."""
        # Acquire load_lock to ensure only one model loads at a time
        await self.load_lock.acquire()
        self.currently_loading = model_alias

        async with self.lock:
            vram_info = self.get_current_vram_info()

            self.model_tracks[model_alias] = ModelVRAMTracking(
                model_alias=model_alias,
                port=port,
                vram_before_load_used_mb=vram_info["used_mb"],
                load_start_time=datetime.now(),
                status="loading"
            )

            logger.info(
                f"[VRAM Tracker] Started tracking '{model_alias}' (load_lock acquired) | "
                f"Total VRAM used before load: {vram_info['used_mb']:.0f} MB | Free VRAM: {vram_info['free_mb']:.0f} MB"
            )

    async def track_model_load_complete(self, model_alias: str):
        """Mark model load as complete. Releases load_lock."""
        async with self.lock:
            if model_alias not in self.model_tracks:
                logger.warning(
                    f"[VRAM Tracker] Model '{model_alias}' not in tracking.")
                # Still release lock if we hold it
                if self.currently_loading == model_alias:
                    self.currently_loading = None
                    if self.load_lock.locked():
                        self.load_lock.release()
                return

            track = self.model_tracks[model_alias]
            vram_info = self.get_current_vram_info()

            # Hitung VRAM yang dipakai: setelah - sebelum
            track.vram_after_load_used_mb = vram_info["used_mb"]
            vram_consumed_mb = track.vram_after_load_used_mb - track.vram_before_load_used_mb

            # Update tracking
            track.current_vram_used_mb = max(0, vram_consumed_mb)
            track.load_end_time = datetime.now()
            track.status = "loaded"

            # Add initial snapshot
            snapshot = ModelVRAMSnapshot(
                model_alias=model_alias,
                timestamp=datetime.now(),
                vram_used_mb=track.current_vram_used_mb,
                total_vram_used_mb=vram_info["used_mb"],
                port=track.port,
                status="loaded"
            )
            track.add_snapshot(snapshot)

            load_duration = track.get_load_duration_sec()

            logger.info(
                f"[VRAM Tracker] '{model_alias}' loaded | VRAM used: {track.current_vram_used_mb:.0f} MB "
                f"({track.current_vram_used_mb / 1024:.2f} GB) | Load time: {load_duration:.1f}s | "
                f"Total VRAM used: {vram_info['used_mb']:.0f} MB | Free: {vram_info['free_mb']:.0f} MB"
            )

        # Release load_lock after updating tracking
        if self.currently_loading == model_alias:
            self.currently_loading = None
            if self.load_lock.locked():
                self.load_lock.release()
                logger.debug(
                    f"[VRAM Tracker] load_lock released for '{model_alias}'")

    async def track_model_load_failed(self, model_alias: str, error: str):
        """Mark model load as failed. Releases load_lock."""
        async with self.lock:
            if model_alias in self.model_tracks:
                logger.warning(
                    f"[VRAM Tracker] '{model_alias}' failed to load: {error}"
                )

                # Hapus dari tracking karena model tidak ter-load
                del self.model_tracks[model_alias]
                logger.info(
                    f"[VRAM Tracker] '{model_alias}' removed from tracking (failed to load)"
                )

        # Release load_lock
        if self.currently_loading == model_alias:
            self.currently_loading = None
            if self.load_lock.locked():
                self.load_lock.release()
                logger.debug(
                    f"[VRAM Tracker] load_lock released for '{model_alias}' (failed)")

    async def track_model_eject(self, model_alias: str):
        async with self.lock:
            if model_alias in self.model_tracks:
                track = self.model_tracks[model_alias]

                logger.info(
                    f"[VRAM Tracker] '{model_alias}' ejected | Was using: {track.current_vram_used_mb:.0f} MB - ({track.current_vram_used_mb / 1024:.2f} GB)"
                )

                del self.model_tracks[model_alias]
            else:
                logger.debug(
                    f"[VRAM Tracker] Model '{model_alias}' not in tracking"
                )

    def get_available_vram_mb(self) -> float:
        """Get actual available VRAM in MB from GPU."""
        vram_info = self.get_current_vram_info()
        return vram_info["free_mb"]

    def can_load_model(self, estimated_vram_mb: float, safety_buffer_mb: float = 200) -> tuple:
        """
        Check if there's enough VRAM to load a new model.

        Args:
            estimated_vram_mb: Estimated VRAM needed for the model
            safety_buffer_mb: Extra buffer to prevent GPU OOM (default 200MB)

        Returns:
            tuple: (can_load: bool, available_mb: float, message: str)
        """
        available_mb = self.get_available_vram_mb()
        required_mb = estimated_vram_mb + safety_buffer_mb

        can_load = available_mb >= required_mb

        if can_load:
            message = f"VRAM OK: {available_mb:.0f} MB available, need {required_mb:.0f} MB (estimated {estimated_vram_mb:.0f} + buffer {safety_buffer_mb:.0f})"
        else:
            message = f"VRAM insufficient: {available_mb:.0f} MB available, need {required_mb:.0f} MB"

        return can_load, available_mb, message

    async def update_all_tracks(self):
        async with self.lock:
            if not self.model_tracks:
                return

            vram_info = self.get_current_vram_info()

            for alias, track in self.model_tracks.items():
                if track.status == "loaded":
                    # Add snapshot
                    snapshot = ModelVRAMSnapshot(
                        model_alias=alias,
                        timestamp=datetime.now(),
                        vram_used_mb=track.current_vram_used_mb,
                        total_vram_used_mb=vram_info["used_mb"],
                        port=track.port,
                        status=track.status
                    )
                    track.add_snapshot(snapshot)

    async def monitor_loop(self):
        """Background monitoring loop."""
        logger.info("[VRAM Tracker] Monitor started")

        while self.monitoring:
            try:
                await self.update_all_tracks()
                await asyncio.sleep(self.monitor_interval_sec)
            except asyncio.CancelledError:
                logger.info("[VRAM Tracker] Monitor cancelled")
                break
            except Exception as e:
                logger.exception(f"[VRAM Tracker] Error in monitor loop: {e}")
                await asyncio.sleep(self.monitor_interval_sec)

        logger.info("[VRAM Tracker] Monitor stopped")

    def start_monitoring(self):
        """Start background monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_task = asyncio.create_task(self.monitor_loop())
            logger.info("[VRAM Tracker] Monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            logger.info("[VRAM Tracker] Monitoring stopped")

    def get_vram_report(self) -> Dict:
        vram_info = self.get_current_vram_info()

        # Calculate total allocated by tracked models
        total_allocated = sum(
            track.current_vram_used_mb
            for track in self.model_tracks.values()
            if track.status == "loaded"
        )

        # Model details
        models_info = []
        for alias, track in self.model_tracks.items():
            model_info = {
                "model_alias": alias,
                "port": track.port,
                "status": track.status,
                "vram_used_mb": round(track.current_vram_used_mb, 2),
                "vram_used_gb": round(track.current_vram_used_mb / 1024, 2),
                "average_usage_mb": round(track.get_average_usage_mb(), 2),
                "load_duration_sec": track.get_load_duration_sec()
            }

            # Add percentage
            if vram_info["total_mb"] > 0:
                model_info["vram_percentage"] = round(
                    (track.current_vram_used_mb /
                     vram_info["total_mb"]) * 100, 2
                )

            models_info.append(model_info)

        # Sort by VRAM usage (descending)
        models_info.sort(key=lambda x: x["vram_used_mb"], reverse=True)

        # Calculate estimated free VRAM for new model
        estimated_free = vram_info["free_mb"]

        # Check if can load more models (use configurable threshold)
        can_load_more = estimated_free >= self.min_vram_required

        return {
            "gpu_info": {
                "total_mb": round(vram_info["total_mb"], 2),
                "total_gb": round(vram_info["total_mb"] / 1024, 2),
                "used_mb": round(vram_info["used_mb"], 2),
                "used_gb": round(vram_info["used_mb"] / 1024, 2),
                "free_mb": round(vram_info["free_mb"], 2),
                "free_gb": round(vram_info["free_mb"] / 1024, 2),
                "usage_percentage": round((vram_info["used_mb"] / vram_info["total_mb"]) * 100, 2) if vram_info["total_mb"] > 0 else 0,
                "baseline_used_mb": round(self.baseline_vram_used_mb, 2)
            },
            "tracked_models_count": len(self.model_tracks),
            "loaded_models_count": len([t for t in self.model_tracks.values() if t.status == "loaded"]),
            "total_allocated_by_models_mb": round(total_allocated, 2),
            "total_allocated_by_models_gb": round(total_allocated / 1024, 2),
            "models": models_info,
            "can_load_more": can_load_more,
            "estimated_free_for_new_model_mb": round(estimated_free, 2),
            "estimated_free_for_new_model_gb": round(estimated_free / 1024, 2),
            "status": self._get_vram_status(vram_info["free_mb"], self.min_vram_required)
        }

    def _get_vram_status(self, free_mb: float, min_required: float) -> str:
        if free_mb >= min_required * 2:
            return "healthy"
        elif free_mb >= min_required:
            return "warning"
        else:
            return "critical"

    def shutdown(self):
        """Shutdown NVML."""
        try:
            pynvml.nvmlShutdown()
            logger.info("[VRAM Tracker] NVML shutdown complete")
        except Exception as e:
            logger.error(f"[VRAM Tracker] NVML shutdown error: {e}")
