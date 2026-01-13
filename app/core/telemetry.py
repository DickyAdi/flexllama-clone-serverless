"""
Telemetry Collection Module

Modul ini menyediakan sistem pengumpulan telemetry untuk tracking performa request
dan menghasilkan summary statistics per model.

Components:
    - RequestMetrics: Dataclass untuk menyimpan metrics satu request
    - TelemetryCollector: Collector dan aggregator untuk telemetry data

Metrics yang di-track per request:
    - request_id: ID unik untuk tracking
    - model_alias: Model yang digunakan
    - endpoint: Endpoint yang dipanggil
    - start_time/end_time: Timestamp untuk duration calculation
    - status_code: HTTP status code response
    - error: Error message jika ada
    - queue_time: Waktu tunggu di queue
    - processing_time: Waktu processing oleh model
    - tokens_generated: Jumlah token yang di-generate

Summary Statistics:
    - Total requests, success rate, error rate
    - Response time stats (avg, min, max, p50, p95)
    - Per-model breakdown dengan detail metrics

Usage:
    telemetry = TelemetryCollector(window_size=1000)
    
    # Record metrics
    await telemetry.record_request(RequestMetrics(
        request_id="req-123",
        model_alias="qwen3-8b",
        endpoint="/v1/chat/completions",
        start_time=time.time(),
        ...
    ))
    
    # Get summary
    summary = telemetry.get_summary()

Note:
    Window size menentukan berapa banyak request terakhir yang disimpan
    untuk perhitungan statistics. Default 1000 requests.
"""

import asyncio
import statistics
from collections import deque
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    """Metrics untuk satu request."""
    request_id: str
    model_alias: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    queue_time: float = 0.0
    processing_time: float = 0.0
    tokens_generated: int = 0

    @property
    def total_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0


class TelemetryCollector:
    """Collect dan aggregate telemetry data."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.recent_requests: deque[RequestMetrics] = deque(maxlen=window_size)
        self.lock = asyncio.Lock()

        # Aggregated metrics per model
        self.model_stats: Dict[str, Dict] = {}

    async def record_request(self, metrics: RequestMetrics):
        """Record request metrics."""
        async with self.lock:
            self.recent_requests.append(metrics)

            # Update model stats
            if metrics.model_alias not in self.model_stats:
                self.model_stats[metrics.model_alias] = {
                    "total_requests": 0,
                    "total_errors": 0,
                    "total_tokens": 0,
                    "response_times": deque(maxlen=100)
                }

            stats = self.model_stats[metrics.model_alias]
            stats["total_requests"] += 1

            if metrics.error:
                stats["total_errors"] += 1

            if metrics.tokens_generated:
                stats["total_tokens"] += metrics.tokens_generated

            if metrics.total_time > 0:
                stats["response_times"].append(metrics.total_time)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.recent_requests:
            return {"message": "No requests recorded yet"}

        # Calculate overall stats
        total_requests = len(self.recent_requests)
        successful = sum(1 for r in self.recent_requests if not r.error)
        failed = total_requests - successful

        response_times = [
            r.total_time for r in self.recent_requests if r.total_time > 0]

        summary = {
            "total_requests": total_requests,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / total_requests * 100):.2f}%",
        }

        if response_times:
            summary["response_time_stats"] = {
                "avg": f"{statistics.mean(response_times):.3f}s",
                "min": f"{min(response_times):.3f}s",
                "max": f"{max(response_times):.3f}s",
                "p50": f"{statistics.median(response_times):.3f}s",
            }

            # Calculate p95 dengan handling untuk < 20 samples
            if len(response_times) >= 20:
                summary["response_time_stats"][
                    "p95"] = f"{statistics.quantiles(response_times, n=20)[18]:.3f}s"
            elif len(response_times) >= 2:
                # Use available data untuk estimate p95
                sorted_times = sorted(response_times)
                p95_index = int(len(sorted_times) * 0.95)
                summary["response_time_stats"][
                    "p95"] = f"{sorted_times[p95_index]:.3f}s (estimated)"
            else:
                summary["response_time_stats"]["p95"] = f"{max(response_times):.3f}s (insufficient data)"

        # Calculate per model stats dengan filter untuk exclude "unknown"
        summary["per_model"] = {}
        for model, stats in self.model_stats.items():
            # Skip jika model_alias adalah "unknown" (error cases)
            if model == "unknown":
                continue

            response_times = list(stats["response_times"])

            model_summary = {
                "total_requests": stats["total_requests"],
                "total_errors": stats["total_errors"],
                "error_rate": f"{(stats['total_errors'] / stats['total_requests'] * 100):.2f}%",
                "total_tokens": stats["total_tokens"],
            }

            if response_times:
                model_summary["avg_response_time"] = f"{statistics.mean(response_times):.3f}s"
                model_summary["min_response_time"] = f"{min(response_times):.3f}s"
                model_summary["max_response_time"] = f"{max(response_times):.3f}s"

                # Add p95 untuk per-model juga
                if len(response_times) >= 20:
                    model_summary[
                        "p95_response_time"] = f"{statistics.quantiles(response_times, n=20)[18]:.3f}s"
                elif len(response_times) >= 2:
                    sorted_times = sorted(response_times)
                    p95_index = int(len(sorted_times) * 0.95)
                    model_summary["p95_response_time"] = f"{sorted_times[p95_index]:.3f}s (est)"
            else:
                model_summary["avg_response_time"] = "N/A"

            summary["per_model"][model] = model_summary

        # Add note jika ada "unknown" entries
        if "unknown" in self.model_stats:
            unknown_count = self.model_stats["unknown"]["total_requests"]
            summary["_note"] = f"{unknown_count} requests with unidentified model (likely errors or monitoring endpoints)"

        return summary
