"""Immutable result models for unattended system validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Mapping


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"


JsonScalar = str | int | float | bool | None


class FrozenArray(tuple):
    """Tuple-backed JSON array preserving its type even when empty."""


class FrozenObject(tuple):
    """Tuple-backed JSON object preserving its type even when empty."""


FrozenJson = JsonScalar | FrozenArray | FrozenObject
Evidence = FrozenObject


def freeze_json(value: Any) -> FrozenJson:
    """Validate and recursively freeze a JSON-compatible value."""
    if isinstance(value, (FrozenArray, FrozenObject)):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON evidence numbers must be finite")
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON evidence object keys must be strings")
            items.append((key, freeze_json(item)))
        return FrozenObject(items)
    if isinstance(value, (list, tuple)):
        return FrozenArray(freeze_json(item) for item in value)
    raise ValueError(f"value of type {type(value).__name__} is not JSON-compatible")


def freeze_evidence(value: Mapping[str, Any] | Evidence | None = None) -> Evidence:
    if isinstance(value, FrozenObject):
        return value
    frozen = freeze_json({} if value is None else value)
    if not isinstance(frozen, FrozenObject):
        raise ValueError("evidence must be a JSON object")
    return frozen


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _reasons(value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("reasons must be a non-empty tuple of non-empty strings")


@dataclass(frozen=True)
class ValidationScenarioResult:
    scenario_id: str
    status: ValidationStatus
    reasons: tuple[str, ...]
    started_at: datetime
    finished_at: datetime
    evidence: Evidence = ()
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        _text(self.scenario_id, "scenario_id")
        if not isinstance(self.status, ValidationStatus):
            raise ValueError("status must be a ValidationStatus")
        _reasons(self.reasons)
        _utc(self.started_at, "started_at")
        _utc(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        frozen = freeze_evidence(dict(self.evidence))
        object.__setattr__(self, "evidence", frozen)
        object.__setattr__(self, "duration_seconds", (self.finished_at - self.started_at).total_seconds())


@dataclass(frozen=True)
class ValidationRunResult:
    run_id: str
    suite: str
    started_at: datetime
    finished_at: datetime
    overall_status: ValidationStatus
    scenario_results: tuple[ValidationScenarioResult, ...]
    environment_summary: Evidence
    cleanup_verified: bool
    report_paths: tuple[str, ...] = ()
    interrupted: bool = False
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        _text(self.suite, "suite")
        _utc(self.started_at, "started_at")
        _utc(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if not isinstance(self.overall_status, ValidationStatus):
            raise ValueError("overall_status must be a ValidationStatus")
        if not isinstance(self.scenario_results, tuple):
            raise ValueError("scenario_results must be a tuple")
        object.__setattr__(self, "environment_summary", freeze_evidence(dict(self.environment_summary)))
        if not isinstance(self.cleanup_verified, bool) or not isinstance(self.interrupted, bool):
            raise ValueError("cleanup_verified and interrupted must be bools")
        if not isinstance(self.report_paths, tuple) or any(not isinstance(path, str) or not path for path in self.report_paths):
            raise ValueError("report_paths must be a tuple of non-empty strings")
        object.__setattr__(self, "duration_seconds", (self.finished_at - self.started_at).total_seconds())


def overall_status(results: tuple[ValidationScenarioResult, ...], cleanup_verified: bool) -> ValidationStatus:
    if not cleanup_verified or any(item.status is ValidationStatus.FAIL for item in results):
        return ValidationStatus.FAIL
    if any(item.status is ValidationStatus.BLOCKED for item in results):
        return ValidationStatus.BLOCKED
    return ValidationStatus.PASS
