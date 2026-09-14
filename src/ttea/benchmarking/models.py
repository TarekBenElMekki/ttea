"""Immutable safety policy and supervision outcomes for future benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math

from ttea.resources import ResourcePolicy


class BenchmarkOutcome(str, Enum):
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    RESOURCE_PRESSURE = "RESOURCE_PRESSURE"
    THERMAL_LIMIT = "THERMAL_LIMIT"
    TELEMETRY_UNAVAILABLE = "TELEMETRY_UNAVAILABLE"
    CANCELLED = "CANCELLED"
    WORKER_FAILURE = "WORKER_FAILURE"


def _positive_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True)
class BenchmarkSafetyPolicy:
    telemetry_interval_seconds: float = 1.0
    consecutive_resource_violation_samples: int = 3
    request_timeout_seconds: float = 120.0
    model_timeout_seconds: float = 600.0
    cooldown_timeout_seconds: float = 10.0
    gpu_warning_temperature_c: float = 70.0
    gpu_abort_temperature_c: float = 80.0
    max_context_tokens: int = 4096
    max_output_tokens: int = 512
    cpu_thread_limit: int = 4

    def __post_init__(self) -> None:
        for name in ("telemetry_interval_seconds", "request_timeout_seconds", "model_timeout_seconds", "cooldown_timeout_seconds", "gpu_warning_temperature_c", "gpu_abort_temperature_c"):
            _positive_number(getattr(self, name), name)
        for name in ("consecutive_resource_violation_samples", "max_context_tokens", "max_output_tokens", "cpu_thread_limit"):
            _positive_integer(getattr(self, name), name)
        if self.gpu_warning_temperature_c >= self.gpu_abort_temperature_c:
            raise ValueError("gpu_warning_temperature_c must be below gpu_abort_temperature_c")
        if self.max_context_tokens > 4096:
            raise ValueError("max_context_tokens must not exceed 4096")
    def validate_with(self, resource_policy: ResourcePolicy) -> None:
        if self.gpu_abort_temperature_c > resource_policy.max_gpu_temperature_c:
            raise ValueError("gpu_abort_temperature_c must not exceed the resource policy temperature threshold")


@dataclass(frozen=True)
class BenchmarkResult:
    outcome: BenchmarkOutcome
    reasons: tuple[str, ...]
    started_at: datetime
    finished_at: datetime
    worker_pid: int | None = None
    worker_result: object | None = None
    worker_started: bool = False
    timeout_kind: str | None = None
    worker_started_epoch: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, BenchmarkOutcome):
            raise ValueError("outcome must be a BenchmarkOutcome")
        if not isinstance(self.reasons, tuple) or not self.reasons or any(not isinstance(item, str) or not item.strip() for item in self.reasons):
            raise ValueError("reasons must be a non-empty tuple of non-empty strings")
        _utc(self.started_at, "started_at")
        _utc(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.worker_pid is not None and (not isinstance(self.worker_pid, int) or isinstance(self.worker_pid, bool) or self.worker_pid <= 0):
            raise ValueError("worker_pid must be a positive integer when present")
        if not isinstance(self.worker_started, bool):
            raise ValueError("worker_started must be a bool")
        if self.timeout_kind not in (None, "request", "model"):
            raise ValueError("timeout_kind must be None, 'request', or 'model'")
        if self.timeout_kind is not None and self.outcome is not BenchmarkOutcome.TIMEOUT:
            raise ValueError("timeout_kind is valid only for a TIMEOUT outcome")
        if self.timeout_kind == "model" and not self.worker_started:
            raise ValueError("a model timeout requires a started worker")
        if self.worker_started_epoch is not None:
            _positive_number(self.worker_started_epoch, "worker_started_epoch")
            if self.worker_pid is None:
                raise ValueError("worker_started_epoch requires worker_pid")
