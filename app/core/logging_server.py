"""
Structured Logging Configuration Module

Modul ini menyediakan konfigurasi logging dengan format terstruktur (JSON)
untuk memudahkan parsing dan analisis log.

Features:
    - Structured JSON format untuk machine-readable logs
    - Human-readable format sebagai fallback
    - Console dan file handlers
    - Suppression untuk noisy libraries (httpx, httpcore)

Log Fields (Structured Mode):
    - timestamp: ISO format timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - module: Python module name
    - function: Function name
    - line: Line number
    - model_alias: (optional) Model yang terkait
    - port: (optional) Port runner
    - status: (optional) Status model
    - exception: (optional) Exception traceback

Log Files:
    - logs/api-gateway.log: Main application logs
    - logs/runners/<alias>_<port>.log: Per-runner logs

Usage:
    from app.core.logging_server import setup_logging
    
    # Setup dengan structured format
    setup_logging(log_level=logging.INFO, use_structured=True)
    
    # Setup dengan simple format
    setup_logging(log_level=logging.DEBUG, use_structured=False)
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter untuk structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Tambahkan extra fields jika ada
        if hasattr(record, 'model_alias'):
            log_data['model_alias'] = record.model_alias
        if hasattr(record, 'port'):
            log_data['port'] = record.port
        if hasattr(record, 'status'):
            log_data['status'] = record.status

        # Tambahkan exception info jika ada
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Setup logging dengan structured format
def setup_logging(log_level=logging.INFO, use_structured=True):
    """Setup logging configuration."""
    if use_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler - buat directory dulu jika belum ada
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'api-gateway.log')
    file_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress httpx di console tapi tetap log ke file
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
