import time
import asyncio
from enum import Enum
from collections import deque
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class RequestPriority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class QueuedRequest:
    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    model_alias: str = field(compare=False)
    body: Dict[Any, Any] = field(compare=False)
    response_future: asyncio.Future = field(compare=False)

    def __post_init__(self):
        # Untuk proper sorting (lower number = higher priority)
        self.sort_key = (self.priority, self.timestamp)


class ModelRequestQueue:
    """Queue per model dengan priority dan backpressure."""

    def __init__(self, model_alias: str, max_queue_size: int = 100):
        self.model_alias = model_alias
        self.max_queue_size = max_queue_size
        self.queue: deque[QueuedRequest] = deque()
        self.processing = False
        self.lock = asyncio.Lock()
        self.queue_not_empty = asyncio.Event()

        # Metrics
        self.total_requests = 0
        self.total_processed = 0
        self.total_rejected = 0

    async def enqueue(
        self,
        request_id: str,
        body: Dict[Any, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 300
    ) -> Dict[Any, Any]:
        """Add request to queue dan tunggu hasil."""

        async with self.lock:
            # Check queue capacity
            if len(self.queue) >= self.max_queue_size:
                self.total_rejected += 1
                raise RuntimeError(
                    f"Queue untuk model '{self.model_alias}' penuh. "
                    f"Coba lagi nanti atau gunakan model lain."
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

            # Insert dengan priority (lower priority value = higher priority)
            inserted = False
            for i, existing_req in enumerate(self.queue):
                if queued_req.sort_key < existing_req.sort_key:
                    self.queue.insert(i, queued_req)
                    inserted = True
                    break

            if not inserted:
                self.queue.append(queued_req)

            self.total_requests += 1
            self.queue_not_empty.set()

        # Wait untuk response dengan timeout
        try:
            result = await asyncio.wait_for(response_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Remove dari queue jika timeout
            async with self.lock:
                try:
                    self.queue.remove(queued_req)
                except ValueError:
                    pass  # Already processed
            raise TimeoutError(f"Request timeout setelah {timeout} detik")

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get next request from queue."""
        async with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_length": len(self.queue),
            "total_requests": self.total_requests,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "processing": self.processing
        }


class QueueManager:
    """Manage queues untuk semua models."""

    def __init__(self, config):
        self.config = config
        self.queues: Dict[str, ModelRequestQueue] = {}
        self.lock = asyncio.Lock()

    async def get_queue(self, model_alias: str) -> ModelRequestQueue:
        """Get or create queue untuk model."""
        async with self.lock:
            if model_alias not in self.queues:
                self.queues[model_alias] = ModelRequestQueue(
                    model_alias=model_alias,
                    max_queue_size=100  # Bisa di-config
                )
            return self.queues[model_alias]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats untuk semua queues."""
        return {
            alias: queue.get_stats()
            for alias, queue in self.queues.items()
        }
