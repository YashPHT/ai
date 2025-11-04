import logging
import logging.config
from pathlib import Path

from .settings import settings


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s"
)


class RequestIdFilter(logging.Filter):
    """Injects a request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple injection
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(log_level: str | None = None) -> None:
    """Configure structured logging for the application."""

    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    level = (log_level or settings.environment).upper()
    resolved_level = "DEBUG" if level == "LOCAL" else "INFO"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {
                "()": "ai_rag.core.logging_config.RequestIdFilter",
            }
        },
        "formatters": {
            "standard": {
                "format": DEFAULT_LOG_FORMAT,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["request_id"],
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filters": ["request_id"],
                "filename": log_directory / "application.log",
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": resolved_level,
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger instance."""

    return logging.getLogger(name)
