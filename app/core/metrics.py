"""
Legacy Metrics Storage Module

Modul ini menyediakan penyimpanan metrics dasar untuk tracking request dan model.
Digunakan sebagai legacy metrics sebelum integrasi Prometheus.

Metrics yang di-track:
    - requests_total: Total request per endpoint
    - requests_success: Request sukses per endpoint
    - requests_failed: Request gagal per endpoint
    - request_duration_seconds: Durasi request per endpoint (list untuk percentile calc)
    - models_loaded_total: Total model yang pernah di-load (kumulatif)
    - models_ejected_total: Total model yang pernah di-eject (kumulatif)
    - startup_time: Waktu startup server

Note:
    Untuk production monitoring, gunakan /metrics endpoint yang menggunakan
    PrometheusMetricsCollector dari prometheus_metrics.py. Endpoint /metrics/legacy
    menggunakan metrics dari modul ini.

Usage:
    from app.core.metrics import metrics
    
    # Increment request counter
    metrics["requests_total"]["/v1/chat/completions"] += 1
    
    # Record duration
    metrics["request_duration_seconds"]["/v1/chat/completions"].append(1.5)
"""

from datetime import datetime
from collections import defaultdict


# Global metrics storage - legacy format untuk backward compatibility
metrics = {
    "requests_total": defaultdict(int),        # Total requests per endpoint
    # Successful requests per endpoint
    "requests_success": defaultdict(int),
    "requests_failed": defaultdict(int),       # Failed requests per endpoint
    # Request durations for percentile calculation
    "request_duration_seconds": defaultdict(list),
    "models_loaded_total": 0,                  # Cumulative model loads
    "models_ejected_total": 0,                 # Cumulative model ejects
    "startup_time": datetime.now().isoformat()  # Server startup timestamp
}
