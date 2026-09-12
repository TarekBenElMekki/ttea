"""Persistent, rotating runtime logging for T/TEA."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ttea.config import Settings, load_settings

LOG_FILENAME = "ttea.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
MAX_LOG_BYTES = 1_000_000
BACKUP_COUNT = 5


def setup_logging(settings: Settings | None = None) -> Path:
    """Configure the package logger and return its persistent log file path."""
    resolved_settings = settings or load_settings()
    log_directory = resolved_settings.log_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / LOG_FILENAME

    logger = logging.getLogger("ttea")
    logger.setLevel(resolved_settings.log_level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        log_path, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setLevel(resolved_settings.log_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a named application logger under the shared ``ttea`` namespace."""
    if not name:
        raise ValueError("Logger name must not be empty")
    return logging.getLogger(f"ttea.{name}")
