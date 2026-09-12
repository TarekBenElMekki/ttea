"""Immutable policy request, rule, and decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from .enums import ExecutionPolicyLevel


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


def _freeze_metadata(value: Any) -> Any:
    """Make mappings and sequences independent from caller-owned mutable values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class ActionRequest:
    """An immutable request for a future runtime operation."""

    mission_id: str
    operation_type: str
    description: str
    id: str = field(default_factory=_new_id)
    agent_id: str | None = None
    target: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("id", self.id),
            ("mission_id", self.mission_id),
            ("operation_type", self.operation_type),
            ("description", self.description),
        ):
            _require_text(value, field_name)
        if self.agent_id is not None:
            _require_text(self.agent_id, "agent_id")
        if self.target is not None:
            _require_text(self.target, "target")
        _require_utc(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class PolicyRule:
    """An exact operation-type rule used by the deterministic policy engine."""

    rule_id: str
    operation_type: str
    level: ExecutionPolicyLevel
    reason: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("rule_id", self.rule_id),
            ("operation_type", self.operation_type),
            ("reason", self.reason),
        ):
            _require_text(value, field_name)
        if not isinstance(self.level, ExecutionPolicyLevel):
            raise ValueError("level must be an ExecutionPolicyLevel")


@dataclass(frozen=True)
class PolicyDecision:
    """An immutable result of evaluating one action request."""

    action_request_id: str
    operation_type: str
    level: ExecutionPolicyLevel
    reason: str
    matched_rule_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("action_request_id", self.action_request_id),
            ("operation_type", self.operation_type),
            ("reason", self.reason),
        ):
            _require_text(value, field_name)
        if self.matched_rule_id is not None:
            _require_text(self.matched_rule_id, "matched_rule_id")
        if not isinstance(self.level, ExecutionPolicyLevel):
            raise ValueError("level must be an ExecutionPolicyLevel")

    @property
    def permits_immediate_execution(self) -> bool:
        return self.level is ExecutionPolicyLevel.AUTO

    @property
    def requires_review(self) -> bool:
        return self.level is ExecutionPolicyLevel.REVIEW

    @property
    def requires_approval(self) -> bool:
        return self.level is ExecutionPolicyLevel.APPROVAL

    @property
    def is_forbidden(self) -> bool:
        return self.level is ExecutionPolicyLevel.FORBIDDEN
