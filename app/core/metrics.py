"""
Global metrics storage untuk tracking sistem.
"""
from datetime import datetime
from collections import defaultdict


# Global metrics storage
metrics = {
    "requests_total": defaultdict(int),
    "requests_success": defaultdict(int),
    "requests_failed": defaultdict(int),
    "request_duration_seconds": defaultdict(list),
    "models_loaded_total": 0,
    "models_ejected_total": 0,
    "startup_time": datetime.now().isoformat()
}
