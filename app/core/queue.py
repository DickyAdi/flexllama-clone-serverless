"""
Request Queue Management Module

Modul ini menyediakan sistem antrian request dengan priority ordering dan backpressure
untuk menangani request ke model LLM secara efisien.

Components:
    - RequestPriority: Enum untuk level prioritas request (HIGH, NORMAL, LOW)
    - QueuedRequest: Dataclass untuk menyimpan request dalam antrian
    - ModelRequestQueue: Queue per-model dengan priority heap dan backpressure
    - QueueManager: Manager untuk semua queue, satu queue per model

Features:
    - Priority-based ordering: Request HIGH diproses duluan
    - Backpressure: Queue memiliki max size untuk mencegah memory exhaustion
    - Timeout handling: Request yang terlalu lama di queue akan timeout
    - Heap-based: O(log n) insertion dan extraction

Flow:
    1. Request masuk via enqueue() dengan priority
    2. Queue processor mengambil request via dequeue() (priority order)
    3. Request diproses dan result dikembalikan via Future
    4. Jika queue penuh, request ditolak dengan error

Usage:
    queue_manager = QueueManager(config)
    queue = await queue_manager.get_queue("qwen3-8b")
    
    # Enqueue dengan priority
    result = await queue.enqueue(
        request_id="req-123",
        body={"prompt": "..."},
        priority=RequestPriority.HIGH,
        timeout=300
    )

Priority Levels:
    - HIGH (1): Request prioritas tinggi (VIP users, urgent tasks)
    - NORMAL (2): Request normal (default)
    - LOW (3): Request prioritas rendah (batch processing, background tasks)
"""

import time
import heapq
import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class QueuedRequest:
    """Request in queue with priority ordering.

    Uses (priority, timestamp) as sort key for heap operations.
    Lower priority value = higher priority (HIGH=1 > NORMAL=2 > LOW=3).
    """
    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    model_alias: str = field(compare=False)
    body: Dict[Any, Any] = field(compare=False)
    response_future: asyncio.Future = field(compare=False)

    def __post_init__(self):
        self.sort_key = (self.priority, self.timestamp)


class ModelRequestQueue:
    """Queue per model dengan priority heap dan backpressure.

    Optimized with heapq for O(log n) insertion instead of O(n) deque insertion.
    """

    def __init__(self, model_alias: str, max_queue_size: int = 100):
        self.model_alias = model_alias
        self.max_queue_size = max_queue_size

        # Use list for heapq operations - O(log n) insert vs O(n) deque insert
        self._heap: List[QueuedRequest] = []
        self.processing = False
        self.lock = asyncio.Lock()
        self.queue_not_empty = asyncio.Event()

        # Metrics
        self.total_requests = 0
        self.total_processed = 0
        self.total_rejected = 0
        self.current_processing = 0

    @property
    def queue(self) -> List[QueuedRequest]:
        """Compatibility property for existing code that accesses .queue"""
        return self._heap

    async def enqueue(
        self,
        request_id: str,
        body: Dict[Any, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 600
    ) -> Dict[Any, Any]:
        """Add request to queue and wait for result."""

        enqueue_time = time.time()

        async with self.lock:
            # Check queue capacity
            if len(self._heap) >= self.max_queue_size:
                self.total_rejected += 1
                raise RuntimeError(
                    f"Queue for model '{self.model_alias}' is full ({self.max_queue_size}). "
                    f"Try again later or use another model."
                )

            # Create queued request
            response_future = asyncio.Future()
            queued_req = QueuedRequest(
                priority=priority.value,
                timestamp=time.time(),
                request_id=request_id,
                model_alias=self.model_alias,
                body=body,
                response_future=response_future
            )

            # O(log n) heap insertion instead of O(n) deque insert
            heapq.heappush(self._heap, queued_req)

            self.total_requests += 1
            self.queue_not_empty.set()

        # Wait for response with timeout
        try:
            result = await asyncio.wait_for(response_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            wait_time = time.time() - enqueue_time
            logger.error(
                f"[{self.model_alias}] Request {request_id} timeout after {wait_time:.1f}s "
                f"(queue_length={len(self._heap)}, processing={self.current_processing})"
            )
            # Remove from queue if timeout - O(n) but rare
            async with self.lock:
                try:
                    self._heap.remove(queued_req)
                    heapq.heapify(self._heap)  # Restore heap property
                except ValueError:
                    pass  # Already processed
            raise TimeoutError(f"Request timeout after {timeout}s in queue")

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get next request from queue (highest priority first)."""
        async with self.lock:
            if self._heap:
                # O(log n) heap pop
                return heapq.heappop(self._heap)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_length": len(self._heap),
            "total_requests": self.total_requests,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "current_processing": self.current_processing,
            "processing": self.processing
        }


class QueueManager:
    """Manage queues for all models."""

    def __init__(self, config):
        self.config = config
        self.queues: Dict[str, ModelRequestQueue] = {}
        self.lock = asyncio.Lock()

    async def get_queue(self, model_alias: str) -> ModelRequestQueue:
        """Get or create queue for model."""
        async with self.lock:
            if model_alias not in self.queues:
                max_size = self.config.system.max_queue_size_per_model
                self.queues[model_alias] = ModelRequestQueue(
                    model_alias=model_alias,
                    max_queue_size=max_size
                )
            return self.queues[model_alias]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all queues."""
        return {
            alias: queue.get_stats()
            for alias, queue in self.queues.items()
        }
