"""Minimal serializable runtime domain models for T/TEA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from .enums import AgentStatus, ApprovalStatus, EventSeverity, MissionStatus, TaskStatus


def _new_id() -> str:
    """Return a collision-resistant, database-independent identifier."""
    return str(uuid4())


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def to_json_compatible(value: Any) -> Any:
    """Convert supported domain values into standard serializable values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    return value


class SerializableModel:
    """Provide one predictable dictionary representation for core models."""

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(asdict(self))


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_utc_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field_name} must use UTC")


@dataclass
class Mission(SerializableModel):
    id: str = field(default_factory=_new_id)
    title: str = ""
    objective: str = ""
    constraints: list[str] = field(default_factory=list)
    status: MissionStatus = MissionStatus.CREATED
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        _require_nonempty(self.id, "id")
        _require_nonempty(self.title, "title")
        _require_nonempty(self.objective, "objective")
        _require_utc_aware(self.created_at, "created_at")
        _require_utc_aware(self.updated_at, "updated_at")


@dataclass
class Task(SerializableModel):
    mission_id: str
    title: str
    objective: str
    id: str = field(default_factory=_new_id)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    assigned_agent_id: str | None = None
    attempts: int = 0
    max_attempts: int = 1
    result: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        for field_name, value in (("id", self.id), ("mission_id", self.mission_id), ("title", self.title), ("objective", self.objective)):
            _require_nonempty(value, field_name)
        if self.priority < 0:
            raise ValueError("priority must be zero or greater")
        if self.attempts < 0 or self.max_attempts < 1 or self.attempts > self.max_attempts:
            raise ValueError("attempts must be between zero and max_attempts")
        _require_utc_aware(self.created_at, "created_at")
        _require_utc_aware(self.updated_at, "updated_at")


@dataclass
class AgentSpec(SerializableModel):
    mission_id: str
    name: str
    role: str
    objective: str
    id: str = field(default_factory=_new_id)
    status: AgentStatus = AgentStatus.PROPOSED
    tools: list[dict[str, Any]] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    model_requirements: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        for field_name, value in (("id", self.id), ("mission_id", self.mission_id), ("name", self.name), ("role", self.role), ("objective", self.objective)):
            _require_nonempty(value, field_name)
        _require_utc_aware(self.created_at, "created_at")
        _require_utc_aware(self.updated_at, "updated_at")


@dataclass
class Approval(SerializableModel):
    mission_id: str
    approval_type: str
    requested_action: str
    reason: str
    id: str = field(default_factory=_new_id)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=_utc_now)
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        for field_name, value in (("id", self.id), ("mission_id", self.mission_id), ("approval_type", self.approval_type), ("requested_action", self.requested_action), ("reason", self.reason)):
            _require_nonempty(value, field_name)
        _require_utc_aware(self.created_at, "created_at")
        if self.resolved_at is not None:
            _require_utc_aware(self.resolved_at, "resolved_at")


@dataclass
class Event(SerializableModel):
    event_type: str
    mission_id: str
    payload: dict[str, Any]
    id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=_utc_now)
    severity: EventSeverity = EventSeverity.INFO
    task_id: str | None = None
    agent_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (("id", self.id), ("event_type", self.event_type), ("mission_id", self.mission_id)):
            _require_nonempty(value, field_name)
        _require_utc_aware(self.timestamp, "timestamp")
