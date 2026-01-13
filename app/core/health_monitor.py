"""
Model Health Monitoring Module

Modul ini menyediakan sistem monitoring kesehatan untuk model yang sedang aktif.
Health checks dilakukan secara periodik untuk mendeteksi model yang bermasalah
dan melakukan auto-recovery jika diperlukan.

Components:
    - HealthCheckResult: Hasil dari satu health check
    - ModelHealth: Tracking status kesehatan satu model
    - HealthMonitor: Background monitor untuk semua active models

Health Status:
    - healthy: Model responding normally
    - degraded: Model mulai bermasalah (2+ consecutive failures)
    - down: Model tidak responding (5+ consecutive failures, akan di-restart)

Features:
    - Periodic health checks ke /health endpoint llama-server
    - Response time tracking untuk performance monitoring
    - Consecutive failure counting untuk status determination
    - Auto-restart untuk model yang down
    - History tracking untuk uptime calculation

Monitoring Flow:
    1. Monitor loop berjalan setiap check_interval_sec (default 30s)
    2. Untuk setiap active model, kirim GET /health ke runner
    3. Record hasil (is_healthy, response_time, error)
    4. Update consecutive_failures counter
    5. Jika model down (5+ failures), trigger restart

Usage:
    health_monitor = HealthMonitor(manager, check_interval_sec=30)
    health_monitor.start()
    
    # Get health status semua models
    health_status = health_monitor.get_all_health()
    
    # Stop monitoring
    await health_monitor.stop()
"""

import time
import httpx
import asyncio
import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result dari health check."""
    timestamp: datetime
    model_alias: str
    is_healthy: bool
    response_time_ms: float
    error: str = ""


@dataclass
class ModelHealth:
    """Track health status untuk satu model."""
    model_alias: str
    consecutive_failures: int = 0
    last_check: datetime = field(default_factory=datetime.now)
    history: List[HealthCheckResult] = field(default_factory=list)
    max_history: int = 100

    def record_check(self, result: HealthCheckResult):
        """Record health check result."""
        self.last_check = result.timestamp

        if result.is_healthy:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        # Keep limited history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    @property
    def is_degraded(self) -> bool:
        """Check jika model dalam degraded state."""
        return self.consecutive_failures >= 2

    @property
    def is_down(self) -> bool:
        """Check jika model completely down."""
        return self.consecutive_failures >= 5

    def get_stats(self) -> Dict:
        """Get statistics dari history."""
        if not self.history:
            return {}

        recent = self.history[-20:]  # Last 20 checks
        healthy_count = sum(1 for h in recent if h.is_healthy)
        response_times = [h.response_time_ms for h in recent if h.is_healthy]

        return {
            "uptime_percentage": f"{(healthy_count / len(recent) * 100):.1f}%",
            "consecutive_failures": self.consecutive_failures,
            "avg_response_time_ms": f"{sum(response_times) / len(response_times):.1f}" if response_times else "N/A",
            "last_check": self.last_check.isoformat(),
            "status": "healthy" if not self.is_degraded else ("degraded" if not self.is_down else "down")
        }


class HealthMonitor:
    """Monitor health dari active models."""

    def __init__(self, manager, check_interval_sec: int = 30):
        self.manager = manager
        self.check_interval_sec = check_interval_sec
        self.model_health: Dict[str, ModelHealth] = {}
        self.lock = asyncio.Lock()
        self.running = False
        self.monitor_task = None

    async def check_model_health(self, model_alias: str, runner) -> HealthCheckResult:
        """Perform health check untuk satu model."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{runner.url}/health",
                    timeout=5.0
                )

                response_time_ms = (time.time() - start_time) * 1000

                is_healthy = response.status_code == 200

                return HealthCheckResult(
                    timestamp=datetime.now(),
                    model_alias=model_alias,
                    is_healthy=is_healthy,
                    response_time_ms=response_time_ms,
                    error="" if is_healthy else f"Status code: {response.status_code}"
                )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                timestamp=datetime.now(),
                model_alias=model_alias,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error=str(e)
            )

    async def monitor_loop(self):
        """Background loop untuk health monitoring."""
        logger.info("Health monitor started")

        while self.running:
            try:
                # Get active runners
                async with self.manager.lock:
                    active_models = {
                        alias: runner
                        for alias, runner in self.manager.active_runners.items()
                        if runner.is_alive() and runner.status == "ready"
                    }

                # Track models that need restart (detected as down)
                models_to_restart = set()

                # Check health untuk setiap model
                for alias, runner in active_models.items():
                    result = await self.check_model_health(alias, runner)

                    async with self.lock:
                        if alias not in self.model_health:
                            self.model_health[alias] = ModelHealth(
                                model_alias=alias)

                        self.model_health[alias].record_check(result)

                        # Log jika ada masalah
                        if self.model_health[alias].is_degraded:
                            logger.warning(
                                f"Model '{alias}' is degraded. "
                                f"Consecutive failures: {self.model_health[alias].consecutive_failures}"
                            )

                        if self.model_health[alias].is_down:
                            logger.error(
                                f"Model '{alias}' is DOWN. Will attempt restart."
                            )
                            # Mark for restart - actual restart will happen outside lock
                            models_to_restart.add(alias)

                # Perform restarts OUTSIDE of lock to prevent blocking health checks
                for alias in models_to_restart:
                    try:
                        logger.info(f"Attempting restart for '{alias}'")
                        await self.manager.eject_model(alias)
                        await asyncio.sleep(2)
                        await self.manager.get_runner_for_request(alias)
                        logger.info(
                            f"Model '{alias}' successfully restarted")
                    except Exception as e:
                        logger.error(
                            f"Failed to restart model '{alias}': {e}")

                # Clean up health tracking untuk models yang sudah tidak active
                async with self.lock:
                    inactive_models = set(
                        self.model_health.keys()) - set(active_models.keys())
                    for alias in inactive_models:
                        del self.model_health[alias]

            except Exception as e:
                logger.exception(f"Error in health monitor loop: {e}")

            # Wait for next check
            await asyncio.sleep(self.check_interval_sec)

        logger.info("Health monitor stopped")

    def start(self):
        """Start health monitoring."""
        if not self.running:
            self.running = True
            self.monitor_task = asyncio.create_task(self.monitor_loop())
            logger.info("Health monitor task created")

    async def stop(self):
        """Stop health monitoring."""
        if self.running:
            self.running = False
            if self.monitor_task:
                await self.monitor_task
            logger.info("Health monitor stopped")

    def get_all_health(self) -> Dict:
        """Get health status untuk semua models."""
        return {
            alias: health.get_stats()
            for alias, health in self.model_health.items()
        }
