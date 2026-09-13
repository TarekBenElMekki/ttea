"""Immutable, validated contracts for the inert tournament framework."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = 1
MAX_CONCURRENCY = 1
DIMENSIONS = frozenset(("compatibility", "instruction_adherence", "structured_output", "planning", "tool_use", "reasoning", "hallucination_resistance", "stability"))
PERFORMANCE_METRICS = (
    "cold_load_latency_seconds", "warm_latency_seconds",
    "decode_tokens_per_second", "peak_ram_bytes", "peak_vram_bytes",
    "gpu_temperature_c", "cleanup_reliability",
)


class TournamentState(str, Enum):
    CREATED = "CREATED"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class CandidateState(str, Enum):
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    RUNNING = "RUNNING"
    PASSED_ROUND = "PASSED_ROUND"
    ELIMINATED = "ELIMINATED"
    FINALIST = "FINALIST"
    SELECTED = "SELECTED"


class TournamentPhase(str, Enum):
    COMPATIBILITY = "COMPATIBILITY"
    SURVIVAL = "SURVIVAL"
    CAPABILITY = "CAPABILITY"
    FINALIST = "FINALIST"


class AttemptStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    DEFERRED = "DEFERRED"


class EliminationReason(str, Enum):
    RESOURCE_DENIED = "RESOURCE_DENIED"
    LOAD_FAILED = "LOAD_FAILED"
    TIMEOUT = "TIMEOUT"
    THERMAL_ABORT = "THERMAL_ABORT"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    RUNTIME_FAILED = "RUNTIME_FAILED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    POLICY_VIOLATION = "POLICY_VIOLATION"


class SelectionRole(str, Enum):
    PRIMARY = "PRIMARY"
    LOW_RESOURCE_FALLBACK = "LOW_RESOURCE_FALLBACK"
    REASONING_SPECIALIST = "REASONING_SPECIALIST"


def _text(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _integer(value: object, name: str, *, allow_zero: bool = False) -> None:
    lower = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")


def _number(value: object, name: str, *, positive: bool = True) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))


def _utc(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _freeze(mapping: Mapping[str, Any], *, scores: bool = False) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError("value must be a mapping")
    copied = dict(mapping)
    if scores:
        for key, value in copied.items():
            if key not in DIMENSIONS:
                raise ValueError(f"unknown score dimension: {key}")
            _number(value, f"score {key}", positive=False)
            if not 0 <= value <= 1:
                raise ValueError(f"score {key} must be between zero and one")
    if scores:
        return MappingProxyType(copied)
    if any(not isinstance(key, str) for key in copied):
        raise ValueError("metric mapping keys must be strings")
    _validate_standard_metrics(copied)
    return MappingProxyType({key: freeze_json(value) for key, value in copied.items()})


def _validate_standard_metrics(metrics: Mapping[str, Any]) -> None:
    nonnegative_numbers = {
        "cold_load_latency_seconds",
        "warm_latency_seconds",
        "decode_tokens_per_second",
        "peak_ram_bytes",
        "peak_vram_bytes",
    }
    for key in nonnegative_numbers:
        value = metrics.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number or None")
    temperature = metrics.get("gpu_temperature_c")
    if temperature is not None and (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature)):
        raise ValueError("gpu_temperature_c must be a finite number or None")
    reliability = metrics.get("cleanup_reliability")
    if reliability is not None:
        if not isinstance(reliability, bool) and (
            not isinstance(reliability, (int, float))
            or not math.isfinite(reliability)
            or not 0 <= reliability <= 1
        ):
            raise ValueError("cleanup_reliability must be bool or a finite number from zero to one")


def freeze_json(value: Any) -> Any:
    """Defensively copy JSON-safe evidence into immutable containers."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metric numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metric mapping keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise ValueError(f"metric value is not JSON-safe: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Convert frozen metric evidence back to deterministic JSON containers."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


TOURNAMENT_TRANSITIONS = {
    TournamentState.CREATED: {TournamentState.WAITING_FOR_RESOURCES, TournamentState.PROVISIONING, TournamentState.RUNNING, TournamentState.CANCELLED, TournamentState.FAILED},
    TournamentState.WAITING_FOR_RESOURCES: {TournamentState.WAITING_FOR_RESOURCES, TournamentState.PROVISIONING, TournamentState.RUNNING, TournamentState.PAUSED, TournamentState.CANCELLED, TournamentState.INTERRUPTED, TournamentState.FAILED},
    TournamentState.PROVISIONING: {TournamentState.WAITING_FOR_RESOURCES, TournamentState.RUNNING, TournamentState.PAUSED, TournamentState.CANCELLED, TournamentState.INTERRUPTED, TournamentState.FAILED},
    TournamentState.RUNNING: {TournamentState.WAITING_FOR_RESOURCES, TournamentState.PROVISIONING, TournamentState.PAUSED, TournamentState.COMPLETED, TournamentState.CANCELLED, TournamentState.INTERRUPTED, TournamentState.FAILED},
    TournamentState.PAUSED: {TournamentState.WAITING_FOR_RESOURCES, TournamentState.PROVISIONING, TournamentState.RUNNING, TournamentState.CANCELLED, TournamentState.INTERRUPTED, TournamentState.FAILED},
    TournamentState.INTERRUPTED: set(),
    TournamentState.COMPLETED: set(),
    TournamentState.FAILED: set(),
    TournamentState.CANCELLED: set(),
}
CANDIDATE_TRANSITIONS = {
    CandidateState.PENDING: {CandidateState.AVAILABLE, CandidateState.ELIMINATED},
    CandidateState.AVAILABLE: {CandidateState.RUNNING, CandidateState.ELIMINATED},
    CandidateState.RUNNING: {CandidateState.PASSED_ROUND, CandidateState.ELIMINATED},
    CandidateState.PASSED_ROUND: {CandidateState.RUNNING, CandidateState.FINALIST, CandidateState.ELIMINATED},
    CandidateState.FINALIST: {CandidateState.RUNNING, CandidateState.SELECTED, CandidateState.ELIMINATED},
    CandidateState.ELIMINATED: set(),
    CandidateState.SELECTED: set(),
}
ATTEMPT_TRANSITIONS = {
    AttemptStatus.PENDING: {AttemptStatus.RUNNING, AttemptStatus.DEFERRED, AttemptStatus.FAILED, AttemptStatus.INTERRUPTED},
    AttemptStatus.RUNNING: {AttemptStatus.COMPLETED, AttemptStatus.FAILED, AttemptStatus.INTERRUPTED},
    AttemptStatus.COMPLETED: set(),
    AttemptStatus.FAILED: set(),
    AttemptStatus.INTERRUPTED: set(),
    AttemptStatus.DEFERRED: set(),
}


def valid_transition(current: TournamentState, target: TournamentState) -> bool:
    return isinstance(current, TournamentState) and target in TOURNAMENT_TRANSITIONS[current]


def require_transition(current: TournamentState, target: TournamentState) -> None:
    if not valid_transition(current, target):
        raise ValueError("illegal tournament transition")


def require_candidate_transition(current: CandidateState, target: CandidateState) -> None:
    if not isinstance(current, CandidateState) or target not in CANDIDATE_TRANSITIONS[current]:
        raise ValueError("illegal candidate transition")


def resume_interrupted(current: TournamentState) -> TournamentState:
    if current is not TournamentState.INTERRUPTED:
        raise ValueError("explicit resume requires INTERRUPTED")
    return TournamentState.PAUSED


def valid_attempt_transition(current: AttemptStatus, target: AttemptStatus) -> bool:
    return isinstance(current, AttemptStatus) and target in ATTEMPT_TRANSITIONS[current]


def require_attempt_transition(current: AttemptStatus, target: AttemptStatus) -> None:
    if not valid_attempt_transition(current, target):
        raise ValueError("illegal attempt transition")


@dataclass(frozen=True)
class CandidateVariant:
    provider: str
    model: str
    quantization: str | None = None
    parameter_size: str | None = None

    def __post_init__(self) -> None:
        _text(self.provider, "provider")
        _text(self.model, "model")
        _text(self.quantization, "quantization", optional=True)
        _text(self.parameter_size, "parameter_size", optional=True)

    @property
    def canonical_identity(self) -> str:
        return "|".join(value.strip().lower() if value else "-" for value in (self.provider, self.model, self.quantization, self.parameter_size))


@dataclass(frozen=True)
class TournamentCandidate:
    candidate_id: str
    family: str
    variant: CandidateVariant
    estimated_disk_bytes: int | None
    estimated_ram_bytes: int
    estimated_vram_bytes: int
    context_target: int
    purpose: str
    enabled: bool = True
    validation_only: bool = False

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        _text(self.family, "family")
        if not isinstance(self.variant, CandidateVariant):
            raise ValueError("variant must be a CandidateVariant")
        _text(self.purpose, "purpose")
        _integer(self.estimated_ram_bytes, "estimated_ram_bytes", allow_zero=True)
        _integer(self.estimated_vram_bytes, "estimated_vram_bytes", allow_zero=True)
        _integer(self.context_target, "context_target")
        if self.estimated_disk_bytes is not None:
            _integer(self.estimated_disk_bytes, "estimated_disk_bytes", allow_zero=True)
        if not isinstance(self.enabled, bool) or not isinstance(self.validation_only, bool):
            raise ValueError("candidate flags must be bool")

    @property
    def canonical_identity(self) -> str:
        return f"{self.variant.canonical_identity}|{self.family.strip().lower()}"


@dataclass(frozen=True)
class TournamentTask:
    task_id: str
    category: str
    prompt: str
    output_contract: str
    scorer_type: str
    weight: float
    max_context: int
    max_output: int
    timeout_seconds: float
    repetitions: int = 1

    def __post_init__(self) -> None:
        for name in ("task_id", "category", "prompt", "output_contract", "scorer_type"):
            _text(getattr(self, name), name)
        _number(self.weight, "weight")
        _integer(self.max_context, "max_context")
        _integer(self.max_output, "max_output")
        _number(self.timeout_seconds, "timeout_seconds")
        _integer(self.repetitions, "repetitions")


@dataclass(frozen=True)
class TournamentRound:
    round_id: str
    phase: TournamentPhase
    task_ids: tuple[str, ...]
    repetitions: int
    context_limit: int
    output_limit: int
    resource_class: str
    timeout_policy: str
    scoring_policy: str
    elimination_policy: str

    def __post_init__(self) -> None:
        _text(self.round_id, "round_id")
        if not isinstance(self.phase, TournamentPhase):
            raise ValueError("phase must be a TournamentPhase")
        if not isinstance(self.task_ids, tuple) or not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be a non-empty tuple of unique IDs")
        for task_id in self.task_ids:
            _text(task_id, "task_id")
        _integer(self.repetitions, "repetitions")
        _integer(self.context_limit, "context_limit")
        _integer(self.output_limit, "output_limit")
        for name in ("resource_class", "timeout_policy", "scoring_policy", "elimination_policy"):
            _text(getattr(self, name), name)


def task_result_id(candidate_id: str, round_id: str, task_id: str, repetition: int) -> str:
    for value, name in ((candidate_id, "candidate_id"), (round_id, "round_id"), (task_id, "task_id")):
        _text(value, name)
    _integer(repetition, "repetition", allow_zero=True)
    prefix = "|".join(f"{len(value)}:{value}" for value in (candidate_id, round_id, task_id))
    return f"{prefix}|{repetition}"


@dataclass(frozen=True)
class TournamentWorkItem:
    candidate_id: str
    round_id: str
    task_id: str
    repetition: int

    def __post_init__(self) -> None:
        task_result_id(self.candidate_id, self.round_id, self.task_id, self.repetition)

    @property
    def stable_id(self) -> str:
        return task_result_id(self.candidate_id, self.round_id, self.task_id, self.repetition)


@dataclass(frozen=True)
class TaskResult:
    candidate_id: str
    round_id: str
    task_id: str
    repetition: int
    status: AttemptStatus
    dimensions: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    elimination: EliminationReason | None = None

    def __post_init__(self) -> None:
        task_result_id(self.candidate_id, self.round_id, self.task_id, self.repetition)
        if self.status not in {AttemptStatus.COMPLETED, AttemptStatus.FAILED}:
            raise ValueError("TaskResult status must be COMPLETED or FAILED")
        if self.elimination is not None and not isinstance(self.elimination, EliminationReason):
            raise ValueError("invalid elimination")
        object.__setattr__(self, "dimensions", _freeze(self.dimensions, scores=True))
        object.__setattr__(self, "metrics", _freeze(self.metrics))

    @property
    def stable_id(self) -> str:
        return task_result_id(self.candidate_id, self.round_id, self.task_id, self.repetition)


@dataclass(frozen=True)
class CandidateRoundResult:
    candidate_id: str
    round_id: str
    task_result_ids: tuple[str, ...]
    status: AttemptStatus
    dimensions: Mapping[str, float]
    metrics: Mapping[str, Any]
    eligible: bool
    elimination: EliminationReason | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        _text(self.round_id, "round_id")
        if not isinstance(self.task_result_ids, tuple) or len(set(self.task_result_ids)) != len(self.task_result_ids):
            raise ValueError("task_result_ids must be a tuple of unique IDs")
        for result_id in self.task_result_ids:
            _text(result_id, "task_result_id")
        if self.status not in {AttemptStatus.COMPLETED, AttemptStatus.FAILED} or not isinstance(self.eligible, bool):
            raise ValueError("invalid aggregate status or eligibility")
        if self.elimination is not None and not isinstance(self.elimination, EliminationReason):
            raise ValueError("invalid aggregate elimination")
        object.__setattr__(self, "dimensions", _freeze(self.dimensions, scores=True))
        object.__setattr__(self, "metrics", _freeze(self.metrics))


@dataclass(frozen=True)
class TournamentResult:
    candidate_id: str
    dimensions: Mapping[str, float]
    metrics: Mapping[str, Any]
    eligible: bool
    elimination: EliminationReason | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be bool")
        if self.elimination is not None and not isinstance(self.elimination, EliminationReason):
            raise ValueError("invalid elimination")
        object.__setattr__(self, "dimensions", _freeze(self.dimensions, scores=True))
        object.__setattr__(self, "metrics", _freeze(self.metrics))


@dataclass(frozen=True)
class TournamentSelection:
    role: SelectionRole
    candidate_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, SelectionRole):
            raise ValueError("role must be a SelectionRole")
        _text(self.candidate_id, "candidate_id", optional=True)
        _text(self.reason, "reason")


def attempt_id(work_id: str, attempt_number: int) -> str:
    _text(work_id, "work_id")
    _integer(attempt_number, "attempt_number")
    return f"{len(work_id)}:{work_id}|attempt:{attempt_number}"


@dataclass(frozen=True)
class TournamentAttempt:
    attempt_id: str
    work_id: str
    attempt_number: int
    status: AttemptStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_id != attempt_id(self.work_id, self.attempt_number):
            raise ValueError("attempt_id does not match work identity")
        if not isinstance(self.status, AttemptStatus):
            raise ValueError("status must be an AttemptStatus")
        if self.started_at is not None:
            _utc(self.started_at, "started_at")
        if self.finished_at is not None:
            _utc(self.finished_at, "finished_at")
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        _text(self.reason, "reason", optional=True)
        if self.status is AttemptStatus.PENDING and (self.started_at is not None or self.finished_at is not None):
            raise ValueError("pending attempt cannot have timestamps")
        if self.status is AttemptStatus.RUNNING and (self.started_at is None or self.finished_at is not None):
            raise ValueError("running attempt requires only started_at")
        if self.status in {AttemptStatus.COMPLETED, AttemptStatus.FAILED, AttemptStatus.INTERRUPTED} and (self.started_at is None or self.finished_at is None):
            raise ValueError("terminal attempt requires started_at and finished_at")
        if self.status is AttemptStatus.DEFERRED:
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError("deferred attempt cannot have execution timestamps")
            if self.reason is None:
                raise ValueError("deferred attempt requires a reason")

    @classmethod
    def new(cls, work_id: str, attempt_number: int) -> TournamentAttempt:
        return cls(attempt_id(work_id, attempt_number), work_id, attempt_number, AttemptStatus.PENDING)


@dataclass(frozen=True)
class EliminationRecord:
    candidate_id: str
    reason: EliminationReason
    timestamp: datetime
    explanation: str
    round_id: str | None = None
    task_id: str | None = None
    work_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        if not isinstance(self.reason, EliminationReason):
            raise ValueError("reason must be an EliminationReason")
        _utc(self.timestamp, "timestamp")
        _text(self.explanation, "explanation")
        for name in ("round_id", "task_id", "work_id"):
            _text(getattr(self, name), name, optional=True)


@dataclass(frozen=True)
class PersistedTournament:
    schema_version: int
    tournament_id: str
    manifest_digest: str
    manifest_snapshot: str
    state: TournamentState
    created_at: datetime
    updated_at: datetime
    current_phase: TournamentPhase | None = None
    current_round_id: str | None = None
    current_candidate_id: str | None = None
    current_task_id: str | None = None
    current_repetition: int | None = None
    progress_completed: int = 0
    progress_total: int = 0
    resource_wait_started_at: datetime | None = None
    resource_wait_reason: str | None = None
    candidate_states: Mapping[str, CandidateState] = field(default_factory=dict)
    attempts: tuple[TournamentAttempt, ...] = ()
    results: Mapping[str, TaskResult] = field(default_factory=dict)
    round_results: tuple[CandidateRoundResult, ...] = ()
    eliminations: tuple[EliminationRecord, ...] = ()
    selections: tuple[TournamentSelection, ...] = ()
    last_event: str | None = None
    last_reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _text(self.tournament_id, "tournament_id")
        _text(self.manifest_digest, "manifest_digest")
        _text(self.manifest_snapshot, "manifest_snapshot")
        if not isinstance(self.state, TournamentState):
            raise ValueError("state must be a TournamentState")
        _utc(self.created_at, "created_at")
        _utc(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if self.current_phase is not None and not isinstance(self.current_phase, TournamentPhase):
            raise ValueError("current_phase must be a TournamentPhase")
        for name in ("current_round_id", "current_candidate_id", "current_task_id"):
            _text(getattr(self, name), name, optional=True)
        if self.current_repetition is not None:
            _integer(self.current_repetition, "current_repetition", allow_zero=True)
        _integer(self.progress_completed, "progress_completed", allow_zero=True)
        _integer(self.progress_total, "progress_total", allow_zero=True)
        if self.progress_completed > self.progress_total:
            raise ValueError("progress_completed cannot exceed progress_total")
        if self.resource_wait_started_at is not None:
            _utc(self.resource_wait_started_at, "resource_wait_started_at")
        _text(self.resource_wait_reason, "resource_wait_reason", optional=True)
        if (self.resource_wait_started_at is None) != (self.resource_wait_reason is None):
            raise ValueError("resource wait timestamp and reason must appear together")
        states = dict(self.candidate_states)
        if any(not isinstance(value, CandidateState) for value in states.values()):
            raise ValueError("candidate_states contains an invalid state")
        for candidate_id in states:
            _text(candidate_id, "candidate_id")
        attempts = tuple(self.attempts)
        if any(not isinstance(value, TournamentAttempt) for value in attempts) or len({value.attempt_id for value in attempts}) != len(attempts):
            raise ValueError("attempts contains invalid or duplicate data")
        results = dict(self.results)
        if any(not isinstance(value, TaskResult) or key != value.stable_id for key, value in results.items()):
            raise ValueError("results contains an invalid result identity")
        round_results = tuple(self.round_results)
        eliminations = tuple(self.eliminations)
        selections = tuple(self.selections)
        if any(not isinstance(value, CandidateRoundResult) for value in round_results):
            raise ValueError("round_results contains invalid data")
        if any(not isinstance(value, EliminationRecord) for value in eliminations):
            raise ValueError("eliminations contains invalid data")
        if any(not isinstance(value, TournamentSelection) for value in selections):
            raise ValueError("selections contains invalid data")
        if len({value.role for value in selections}) != len(selections):
            raise ValueError("selections must not contain duplicate roles")
        _text(self.last_event, "last_event", optional=True)
        _text(self.last_reason, "last_reason", optional=True)
        object.__setattr__(self, "candidate_states", MappingProxyType(states))
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "results", MappingProxyType(results))
        object.__setattr__(self, "round_results", round_results)
        object.__setattr__(self, "eliminations", eliminations)
        object.__setattr__(self, "selections", selections)


@dataclass(frozen=True)
class TournamentSnapshot:
    tournament_id: str
    state: TournamentState
    phase: TournamentPhase | None
    round_id: str | None
    candidate_id: str | None
    task_id: str | None
    progress_completed: int
    progress_total: int
    resource_wait_reason: str | None
    last_event: str | None
    latest_scores: Mapping[str, float] = field(default_factory=dict)
    latest_elimination: EliminationRecord | None = None

    def __post_init__(self) -> None:
        _text(self.tournament_id, "tournament_id")
        if not isinstance(self.state, TournamentState):
            raise ValueError("state must be a TournamentState")
        _integer(self.progress_completed, "progress_completed", allow_zero=True)
        _integer(self.progress_total, "progress_total", allow_zero=True)
        if self.progress_completed > self.progress_total:
            raise ValueError("invalid snapshot progress")
        object.__setattr__(self, "latest_scores", _freeze(self.latest_scores, scores=True))


def recover_stale_attempts(tournament: PersistedTournament, recovered_at: datetime) -> PersistedTournament:
    """Mark orphaned RUNNING attempts INTERRUPTED without fabricating results."""
    _utc(recovered_at, "recovered_at")
    recovered = tuple(
        replace(item, status=AttemptStatus.INTERRUPTED, finished_at=recovered_at, reason=item.reason or "stale running attempt recovered after restart")
        if item.status is AttemptStatus.RUNNING else item
        for item in tournament.attempts
    )
    if recovered == tournament.attempts:
        return tournament
    if tournament.state is not TournamentState.INTERRUPTED:
        require_transition(tournament.state, TournamentState.INTERRUPTED)
    return replace(tournament, attempts=recovered, state=TournamentState.INTERRUPTED, updated_at=recovered_at, last_event="stale attempts recovered", last_reason="one or more running attempts had no live owner")


def retry_attempt(previous: TournamentAttempt) -> TournamentAttempt:
    """Create a new pending attempt; completed or live work cannot be retried."""
    if previous.status in {AttemptStatus.PENDING, AttemptStatus.RUNNING, AttemptStatus.COMPLETED}:
        raise ValueError(f"cannot retry an attempt in {previous.status.value}")
    return TournamentAttempt.new(previous.work_id, previous.attempt_number + 1)


def snapshot(tournament: PersistedTournament) -> TournamentSnapshot:
    return TournamentSnapshot(
        tournament_id=tournament.tournament_id,
        state=tournament.state,
        phase=tournament.current_phase,
        round_id=tournament.current_round_id,
        candidate_id=tournament.current_candidate_id,
        task_id=tournament.current_task_id,
        progress_completed=tournament.progress_completed,
        progress_total=tournament.progress_total,
        resource_wait_reason=tournament.resource_wait_reason,
        last_event=tournament.last_event,
        latest_scores=tournament.round_results[-1].dimensions if tournament.round_results else {},
        latest_elimination=tournament.eliminations[-1] if tournament.eliminations else None,
    )
