"""Minimal environment-backed settings."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOG_DIRECTORY = Path("logs")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DATABASE_PATH = Path("data") / "ttea.db"


@dataclass(frozen=True)
class Settings:
    """Foundation settings resolved from TTEA_ environment variables."""

    log_directory: Path
    log_level: int
    database_path: Path = DEFAULT_DATABASE_PATH


def load_settings() -> Settings:
    """Load validated foundation settings from the environment."""
    log_directory = Path(os.getenv("TTEA_LOG_DIRECTORY", str(DEFAULT_LOG_DIRECTORY)))
    database_path = Path(os.getenv("TTEA_DATABASE_PATH", str(DEFAULT_DATABASE_PATH)))
    level_name = os.getenv("TTEA_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    log_level = logging.getLevelNamesMapping().get(level_name)
    if not isinstance(log_level, int):
        raise ValueError(f"TTEA_LOG_LEVEL must be a valid logging level, got {level_name!r}")
    return Settings(log_directory=log_directory, log_level=log_level, database_path=database_path)
