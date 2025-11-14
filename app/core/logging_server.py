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
    console_handler = logging.StreamHandler()
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
