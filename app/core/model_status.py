"""
Model Status Tracking Module

Modul ini menyediakan sistem tracking status real-time untuk semua model
dengan dukungan SSE (Server-Sent Events) untuk live updates ke frontend.

Components:
    - ModelStatus: Enum untuk status model (OFF, STARTING, LOADING, READY, dll)
    - ModelStatusInfo: Dataclass dengan info lengkap status model
    - ModelStatusTracker: Main tracker dengan SSE support dan file persistence

Model Status Flow:
    OFF -> STARTING -> LOADING -> READY/LOADED
                    -> CRASHED (jika gagal)
                    -> FAILED (jika config error)
    READY -> STOPPING -> OFF

Status Definitions:
    - OFF: Model tidak aktif, tidak ada runner process
    - STARTING: Subprocess llama-server sedang di-spawn
    - LOADING: Model sedang di-load ke VRAM
    - READY/LOADED: Model siap menerima request
    - STANDBY: Model idle tapi masih di-memory (deprecated)
    - STOPPING: Model sedang dihentikan
    - CRASHED: Model crash unexpectedly
    - FAILED: Model gagal start (configuration/VRAM error)

Features:
    - Real-time status tracking per model
    - SSE broadcasting untuk live updates
    - File persistence untuk access sebelum FastAPI ready
    - VRAM usage tracking
    - Timestamp tracking (started_at, last_used_at, updated_at)

Usage:
    tracker = init_status_tracker()
    await tracker.initialize_from_config(["qwen3-8b", "gemma3-4b"])
    
    # Update status
    await tracker.update_status("qwen3-8b", ModelStatus.LOADING)
    await tracker.update_status("qwen3-8b", ModelStatus.READY, port=8085)
    
    # Get status
    status = await tracker.get_status("qwen3-8b")
    all_statuses = await tracker.get_full_status()
    
    # SSE subscription
    queue = await tracker.subscribe()
    # ... consume updates from queue ...
    await tracker.unsubscribe(queue)
"""

import json
import asyncio
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Enum untuk status model.

    Status yang di-track:
    - off: Model tidak aktif, tidak ada runner
    - starting: Subprocess sedang di-spawn
    - loading: Model sedang di-load ke VRAM
    - loaded/ready: Model siap menerima request
    - standby: Model idle tapi masih di-memory (deprecated, same as ready)
    - stopping: Model sedang dihentikan
    - crashed: Model crash
    - failed: Model gagal start (configuration error)

    """
    OFF = "off"
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    LOADED = "loaded"  # Alias for READY
    STANDBY = "standby"  # Model idle tapi masih di-memory
    STOPPING = "stopping"
    CRASHED = "crashed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ModelStatusInfo:
    """Info lengkap status model."""
    alias: str
    status: ModelStatus
    port: Optional[int] = None
    started_at: Optional[str] = None
    last_used_at: Optional[str] = None
    load_progress: Optional[float] = None  # 0-100%
    error_message: Optional[str] = None
    vram_used_mb: Optional[float] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert ke dictionary."""
        return {
            "alias": self.alias,
            "status": self.status.value if isinstance(self.status, ModelStatus) else self.status,
            "port": self.port,
            "started_at": self.started_at,
            "last_used_at": self.last_used_at,
            "load_progress": self.load_progress,
            "error_message": self.error_message,
            "vram_used_mb": self.vram_used_mb,
            "updated_at": self.updated_at
        }


class ModelStatusTracker:
    """
    Tracker untuk status semua model.

    Features:
    - Track status setiap model
    - Broadcast updates ke SSE subscribers
    - Persist state ke file untuk pre-FastAPI access
    """

    def __init__(self, status_file_path: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            status_file_path: Path ke file untuk persist status.
                            Jika None, akan menggunakan default di PROJECT_ROOT.
        """
        self.statuses: Dict[str, ModelStatusInfo] = {}
        self.lock = asyncio.Lock()

        # SSE subscribers
        self._subscribers: Set[asyncio.Queue] = set()
        self._subscribers_lock = asyncio.Lock()

        # Status file path
        if status_file_path:
            self.status_file = Path(status_file_path)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.status_file = project_root / "logs" / "model_status.json"

        # Ensure parent directory exists
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Server status
        self.server_status = "initializing"
        self.server_started_at: Optional[str] = None

        logger.info(
            f"ModelStatusTracker initialized. Status file: {self.status_file}")

    async def initialize_from_config(self, model_aliases: List[str]):
        """
        Initialize status untuk semua model dari config.
        Semua model mulai dengan status OFF.

        Args:
            model_aliases: List alias model dari config
        """
        async with self.lock:
            for alias in model_aliases:
                if alias not in self.statuses:
                    self.statuses[alias] = ModelStatusInfo(
                        alias=alias,
                        status=ModelStatus.OFF
                    )

            # Save to file
            await self._save_to_file_unsafe()

        logger.info(f"Initialized status for {len(model_aliases)} models")

    async def update_status(
        self,
        alias: str,
        status: ModelStatus,
        port: Optional[int] = None,
        error_message: Optional[str] = None,
        load_progress: Optional[float] = None,
        vram_used_mb: Optional[float] = None
    ):
        """
        Update status model.

        Args:
            alias: Model alias
            status: New status
            port: Port jika running
            error_message: Error message jika failed/crashed
            load_progress: Loading progress 0-100%
            vram_used_mb: VRAM usage in MB
        """
        async with self.lock:
            now = datetime.now().isoformat()

            if alias in self.statuses:
                info = self.statuses[alias]
                info.status = status
                info.updated_at = now

                if port is not None:
                    info.port = port
                if error_message is not None:
                    info.error_message = error_message
                if load_progress is not None:
                    info.load_progress = load_progress
                if vram_used_mb is not None:
                    info.vram_used_mb = vram_used_mb

                # Track timestamps
                if status == ModelStatus.STARTING:
                    info.started_at = now
                    info.error_message = None
                    info.load_progress = 0
                elif status in [ModelStatus.READY, ModelStatus.LOADED]:
                    info.last_used_at = now
                    info.load_progress = 100
                elif status == ModelStatus.OFF:
                    info.port = None
                    info.load_progress = None
                    info.vram_used_mb = None
            else:
                # Create new entry
                self.statuses[alias] = ModelStatusInfo(
                    alias=alias,
                    status=status,
                    port=port,
                    error_message=error_message,
                    load_progress=load_progress,
                    vram_used_mb=vram_used_mb,
                    started_at=now if status == ModelStatus.STARTING else None,
                    updated_at=now
                )

            # Save to file
            await self._save_to_file_unsafe()

        # Broadcast to subscribers (outside lock)
        await self._broadcast_update(alias)

        logger.debug(f"Model '{alias}' status updated to: {status.value}")

    async def update_last_used(self, alias: str):
        """Update last_used_at timestamp."""
        async with self.lock:
            if alias in self.statuses:
                self.statuses[alias].last_used_at = datetime.now().isoformat()
                self.statuses[alias].updated_at = datetime.now().isoformat()

    async def update_vram(self, alias: str, vram_used_mb: float):
        """Update VRAM usage untuk model."""
        async with self.lock:
            if alias in self.statuses:
                self.statuses[alias].vram_used_mb = vram_used_mb
                self.statuses[alias].updated_at = datetime.now().isoformat()

    async def get_status(self, alias: str) -> Optional[ModelStatusInfo]:
        """Get status untuk satu model."""
        async with self.lock:
            return self.statuses.get(alias)

    async def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status semua model."""
        async with self.lock:
            return {
                alias: info.to_dict()
                for alias, info in self.statuses.items()
            }

    async def get_full_status(self) -> Dict[str, Any]:
        """
        Get full status including server info.
        Format yang dikirim ke frontend.
        """
        async with self.lock:
            models = {
                alias: info.to_dict()
                for alias, info in self.statuses.items()
            }

            # Count by status
            status_counts = {}
            for info in self.statuses.values():
                status_val = info.status.value if isinstance(
                    info.status, ModelStatus) else info.status
                status_counts[status_val] = status_counts.get(
                    status_val, 0) + 1

            return {
                "server": {
                    "status": self.server_status,
                    "started_at": self.server_started_at,
                    "updated_at": datetime.now().isoformat()
                },
                "models": models,
                "summary": {
                    "total": len(self.statuses),
                    "by_status": status_counts
                }
            }

    async def set_server_status(self, status: str):
        """Set server status."""
        self.server_status = status
        if status == "ready":
            self.server_started_at = datetime.now().isoformat()

        # Save to file
        async with self.lock:
            await self._save_to_file_unsafe()

        # Broadcast server status change
        await self._broadcast_server_update()

    async def _save_to_file_unsafe(self):
        """
        Save status ke file. MUST be called within lock.
        This allows access before FastAPI is ready.
        """
        try:
            data = {
                "server": {
                    "status": self.server_status,
                    "started_at": self.server_started_at,
                    "updated_at": datetime.now().isoformat()
                },
                "models": {
                    alias: info.to_dict()
                    for alias, info in self.statuses.items()
                }
            }

            # Write atomically
            temp_file = self.status_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Rename (atomic on POSIX)
            temp_file.rename(self.status_file)

        except Exception as e:
            logger.error(f"Failed to save status file: {e}")

    @classmethod
    def read_status_file(cls, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Static method untuk baca status file.
        Bisa dipanggil tanpa instance, useful untuk access sebelum FastAPI ready.

        Args:
            file_path: Path ke status file. Jika None, gunakan default.

        Returns:
            Dict dengan status info, atau empty dict jika file tidak ada.
        """
        if file_path:
            status_file = Path(file_path)
        else:
            project_root = Path(__file__).parent.parent.parent
            status_file = project_root / "logs" / "model_status.json"

        try:
            if status_file.exists():
                with open(status_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to read status file: {e}")
            return {}

    # --- SSE Subscription Methods ---

    async def subscribe(self) -> asyncio.Queue:
        """
        Subscribe ke status updates.

        Returns:
            Queue yang akan receive updates
        """
        queue: asyncio.Queue = asyncio.Queue()

        async with self._subscribers_lock:
            self._subscribers.add(queue)

        logger.debug(f"New SSE subscriber. Total: {len(self._subscribers)}")
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe dari updates."""
        async with self._subscribers_lock:
            self._subscribers.discard(queue)

        logger.debug(
            f"SSE subscriber removed. Total: {len(self._subscribers)}")

    async def _broadcast_update(self, alias: str):
        """Broadcast model status update ke semua subscribers."""
        async with self.lock:
            if alias not in self.statuses:
                return

            data = {
                "type": "model_update",
                "data": self.statuses[alias].to_dict()
            }

        await self._send_to_subscribers(data)

    async def _broadcast_server_update(self):
        """Broadcast server status update."""
        data = {
            "type": "server_update",
            "data": {
                "status": self.server_status,
                "started_at": self.server_started_at,
                "updated_at": datetime.now().isoformat()
            }
        }

        await self._send_to_subscribers(data)

    async def _send_to_subscribers(self, data: Dict[str, Any]):
        """Send data ke semua subscribers."""
        async with self._subscribers_lock:
            dead_subscribers = []

            for queue in self._subscribers:
                try:
                    # Non-blocking put
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    # Queue full, subscriber too slow
                    logger.warning(
                        "SSE subscriber queue full, dropping message")
                except Exception as e:
                    logger.error(f"Error sending to subscriber: {e}")
                    dead_subscribers.append(queue)

            # Remove dead subscribers
            for queue in dead_subscribers:
                self._subscribers.discard(queue)


# Global instance - akan di-init saat startup
_status_tracker: Optional[ModelStatusTracker] = None


def get_status_tracker() -> Optional[ModelStatusTracker]:
    """Get global status tracker instance."""
    return _status_tracker


def init_status_tracker(status_file_path: Optional[str] = None) -> ModelStatusTracker:
    """Initialize global status tracker."""
    global _status_tracker
    _status_tracker = ModelStatusTracker(status_file_path)
    return _status_tracker
