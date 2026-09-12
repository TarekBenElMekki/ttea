"""SQLite connection and schema initialization helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import logging
import sqlite3

from ttea.config import load_settings
from ttea.logging import get_logger, setup_logging

LOGGER = get_logger("persistence.database")
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS missions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    constraints_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE RESTRICT,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    dependencies_json TEXT NOT NULL,
    assigned_agent_id TEXT,
    attempts INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    result_json TEXT,
    artifacts_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    permissions_json TEXT NOT NULL,
    model_requirements_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE RESTRICT,
    approval_type TEXT NOT NULL,
    requested_action TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE RESTRICT,
    task_id TEXT,
    agent_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_mission ON tasks(mission_id);
CREATE INDEX IF NOT EXISTS idx_agents_mission ON agents(mission_id);
CREATE INDEX IF NOT EXISTS idx_approvals_mission ON approvals(mission_id);
CREATE INDEX IF NOT EXISTS idx_events_mission_timestamp ON events(mission_id, timestamp, id);
"""


class Database:
    """Local SQLite database access; callers explicitly initialize its schema."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else load_settings().database_path

    def connect(self) -> sqlite3.Connection:
        """Open a connection with foreign-key enforcement enabled."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        """Create the schema explicitly if it does not already exist."""
        if not logging.getLogger("ttea").handlers:
            setup_logging()
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        LOGGER.info("SQLite database initialized at %s", self.path)

    @contextmanager
    def connection(self):
        """Yield one transactional connection and close it afterward."""
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()
