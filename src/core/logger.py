"""
Structured JSON Logging Module.

Provides enterprise-grade, machine-readable JSON logging for observability,
log aggregation (e.g., Datadog, ELK, CloudWatch), and request correlation.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line serialized JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Include custom fields passed in extra
        if hasattr(record, "request_id"):
            log_payload["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_payload["latency_ms"] = record.latency_ms
        if hasattr(record, "status_code"):
            log_payload["status_code"] = record.status_code
        if hasattr(record, "extra_data"):
            log_payload["extra_data"] = record.extra_data

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, default=str)


def get_logger(name: str = "sentinelml") -> logging.Logger:
    """Factory creating and configuring a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
