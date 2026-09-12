"""Focused SQLite repository facade for T/TEA core models."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from ttea.core import AgentSpec, Approval, Event, Mission, Task, to_json_compatible
from ttea.core.enums import AgentStatus, ApprovalStatus, EventSeverity, MissionStatus, TaskStatus

from .database import Database
from .exceptions import DuplicateEventError


def _json(value: Any) -> str:
    return json.dumps(to_json_compatible(value), sort_keys=True, separators=(",", ":"))


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


class Repository:
    """Persist and reconstruct T/TEA runtime models using a local SQLite database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def initialize(self) -> None:
        """Explicitly initialize this repository's database schema."""
        self.database.initialize()

    def save_mission(self, mission: Mission) -> None:
        self._upsert(
            "missions", ("id", "title", "objective", "constraints_json", "status", "created_at", "updated_at"),
            (mission.id, mission.title, mission.objective, _json(mission.constraints), mission.status.value, mission.created_at.isoformat(), mission.updated_at.isoformat()),
        )

    def get_mission(self, mission_id: str) -> Mission | None:
        row = self._one("SELECT * FROM missions WHERE id = ?", (mission_id,))
        return self._mission(row) if row else None

    def list_missions(self) -> list[Mission]:
        return [self._mission(row) for row in self._many("SELECT * FROM missions ORDER BY created_at, id")]

    def save_task(self, task: Task) -> None:
        self._upsert(
            "tasks", ("id", "mission_id", "title", "objective", "status", "priority", "dependencies_json", "assigned_agent_id", "attempts", "max_attempts", "result_json", "artifacts_json", "created_at", "updated_at"),
            (task.id, task.mission_id, task.title, task.objective, task.status.value, task.priority, _json(task.dependencies), task.assigned_agent_id, task.attempts, task.max_attempts, _json(task.result) if task.result is not None else None, _json(task.artifacts), task.created_at.isoformat(), task.updated_at.isoformat()),
        )

    def get_task(self, task_id: str) -> Task | None:
        row = self._one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return self._task(row) if row else None

    def list_tasks(self, mission_id: str) -> list[Task]:
        return [self._task(row) for row in self._many("SELECT * FROM tasks WHERE mission_id = ? ORDER BY created_at, id", (mission_id,))]

    def save_agent(self, agent: AgentSpec) -> None:
        self._upsert(
            "agents", ("id", "mission_id", "name", "role", "objective", "status", "tools_json", "permissions_json", "model_requirements_json", "created_at", "updated_at"),
            (agent.id, agent.mission_id, agent.name, agent.role, agent.objective, agent.status.value, _json(agent.tools), _json(agent.permissions), _json(agent.model_requirements), agent.created_at.isoformat(), agent.updated_at.isoformat()),
        )

    def get_agent(self, agent_id: str) -> AgentSpec | None:
        row = self._one("SELECT * FROM agents WHERE id = ?", (agent_id,))
        return self._agent(row) if row else None

    def list_agents(self, mission_id: str) -> list[AgentSpec]:
        return [self._agent(row) for row in self._many("SELECT * FROM agents WHERE mission_id = ? ORDER BY created_at, id", (mission_id,))]

    def save_approval(self, approval: Approval) -> None:
        self._upsert(
            "approvals", ("id", "mission_id", "approval_type", "requested_action", "reason", "status", "created_at", "resolved_at"),
            (approval.id, approval.mission_id, approval.approval_type, approval.requested_action, approval.reason, approval.status.value, approval.created_at.isoformat(), approval.resolved_at.isoformat() if approval.resolved_at else None),
        )

    def get_approval(self, approval_id: str) -> Approval | None:
        row = self._one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        return self._approval(row) if row else None

    def list_approvals(self, mission_id: str) -> list[Approval]:
        return [self._approval(row) for row in self._many("SELECT * FROM approvals WHERE mission_id = ? ORDER BY created_at, id", (mission_id,))]

    def append_event(self, event: Event) -> None:
        with self.database.connection() as connection:
            existing = connection.execute("SELECT 1 FROM events WHERE id = ?", (event.id,)).fetchone()
            if existing:
                raise DuplicateEventError(f"Event {event.id} already exists and cannot be overwritten")
            connection.execute(
                "INSERT INTO events (id, timestamp, event_type, severity, mission_id, task_id, agent_id, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (event.id, event.timestamp.isoformat(), event.event_type, event.severity.value, event.mission_id, event.task_id, event.agent_id, _json(event.payload)),
            )

    save_event = append_event

    def get_event(self, event_id: str) -> Event | None:
        row = self._one("SELECT * FROM events WHERE id = ?", (event_id,))
        return self._event(row) if row else None

    def list_events(self, mission_id: str) -> list[Event]:
        return [self._event(row) for row in self._many("SELECT * FROM events WHERE mission_id = ? ORDER BY timestamp, id", (mission_id,))]

    def _upsert(
        self,
        table: str,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
        excluded_update_columns: frozenset[str] = frozenset({"id", "created_at"}),
    ) -> None:
        column_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{column} = excluded.{column}"
            for column in columns
            if column not in excluded_update_columns
        )
        with self.database.connection() as connection:
            connection.execute(f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}", values)

    def _one(self, query: str, values: tuple[Any, ...]) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute(query, values).fetchone()

    def _many(self, query: str, values: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute(query, values).fetchall()

    @staticmethod
    def _mission(row: sqlite3.Row) -> Mission:
        return Mission(id=row["id"], title=row["title"], objective=row["objective"], constraints=json.loads(row["constraints_json"]), status=MissionStatus(row["status"]), created_at=_datetime(row["created_at"]), updated_at=_datetime(row["updated_at"]))

    @staticmethod
    def _task(row: sqlite3.Row) -> Task:
        return Task(id=row["id"], mission_id=row["mission_id"], title=row["title"], objective=row["objective"], status=TaskStatus(row["status"]), priority=row["priority"], dependencies=json.loads(row["dependencies_json"]), assigned_agent_id=row["assigned_agent_id"], attempts=row["attempts"], max_attempts=row["max_attempts"], result=json.loads(row["result_json"]) if row["result_json"] is not None else None, artifacts=json.loads(row["artifacts_json"]), created_at=_datetime(row["created_at"]), updated_at=_datetime(row["updated_at"]))

    @staticmethod
    def _agent(row: sqlite3.Row) -> AgentSpec:
        return AgentSpec(id=row["id"], mission_id=row["mission_id"], name=row["name"], role=row["role"], objective=row["objective"], status=AgentStatus(row["status"]), tools=json.loads(row["tools_json"]), permissions=json.loads(row["permissions_json"]), model_requirements=json.loads(row["model_requirements_json"]), created_at=_datetime(row["created_at"]), updated_at=_datetime(row["updated_at"]))

    @staticmethod
    def _approval(row: sqlite3.Row) -> Approval:
        return Approval(id=row["id"], mission_id=row["mission_id"], approval_type=row["approval_type"], requested_action=row["requested_action"], reason=row["reason"], status=ApprovalStatus(row["status"]), created_at=_datetime(row["created_at"]), resolved_at=_datetime(row["resolved_at"]))

    @staticmethod
    def _event(row: sqlite3.Row) -> Event:
        return Event(id=row["id"], timestamp=_datetime(row["timestamp"]), event_type=row["event_type"], severity=EventSeverity(row["severity"]), mission_id=row["mission_id"], task_id=row["task_id"], agent_id=row["agent_id"], payload=json.loads(row["payload_json"]))
