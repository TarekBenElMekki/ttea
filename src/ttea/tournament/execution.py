"""Persistent contracts for the controlled survival tournament."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from ttea.benchmarking.models import BenchmarkOutcome
from ttea.resources import ResourceDecision, ResourceRequest

from .manifest import TournamentManifest
from .models import CandidateState, PersistedTournament, SCHEMA_VERSION, TournamentState, freeze_json, thaw_json
from .store import TournamentStore, TournamentStoreError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _utc(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _positive(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class SurvivalCampaignState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"


SURVIVAL_STATE_TRANSITIONS = {
    SurvivalCampaignState.CREATED: {
        SurvivalCampaignState.RUNNING,
        SurvivalCampaignState.WAITING_FOR_RESOURCES,
        SurvivalCampaignState.CANCELLED,
        SurvivalCampaignState.INTERRUPTED,
        SurvivalCampaignState.FAILED,
    },
    SurvivalCampaignState.RUNNING: {
        SurvivalCampaignState.WAITING_FOR_RESOURCES,
        SurvivalCampaignState.COMPLETED,
        SurvivalCampaignState.CANCELLED,
        SurvivalCampaignState.INTERRUPTED,
        SurvivalCampaignState.FAILED,
    },
    SurvivalCampaignState.WAITING_FOR_RESOURCES: {
        SurvivalCampaignState.RUNNING,
        SurvivalCampaignState.CANCELLED,
        SurvivalCampaignState.INTERRUPTED,
        SurvivalCampaignState.FAILED,
    },
    SurvivalCampaignState.INTERRUPTED: {
        SurvivalCampaignState.CANCELLED,
    },
    SurvivalCampaignState.COMPLETED: set(),
    SurvivalCampaignState.CANCELLED: set(),
    SurvivalCampaignState.FAILED: set(),
}


ACTIVE_SURVIVAL_STATES = frozenset((
    SurvivalCampaignState.CREATED,
    SurvivalCampaignState.RUNNING,
    SurvivalCampaignState.WAITING_FOR_RESOURCES,
))


class ProvisioningState(str, Enum):
    PENDING = "PENDING"
    INSPECTING = "INSPECTING"
    DOWNLOADING = "DOWNLOADING"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class SurvivalOutcome(str, Enum):
    PENDING = "PENDING"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    ELIMINATED = "ELIMINATED"
    BLOCKED = "BLOCKED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"


class AttemptEvidenceState(str, Enum):
    FINALIZED = "FINALIZED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL_CANDIDATE_OUTCOMES = frozenset((
    SurvivalOutcome.PASSED, SurvivalOutcome.ELIMINATED,
    SurvivalOutcome.BLOCKED, SurvivalOutcome.CANCELLED,
))


@dataclass(frozen=True)
class SurvivalExecutionPolicy:
    allow_model_download: bool = False
    through: str = "survival"
    pull_timeout_seconds: float = 30 * 60
    request_timeout_seconds: float = 300.0
    model_timeout_seconds: float = 180.0
    resource_wait_timeout_seconds: float = 6 * 60 * 60
    resource_retry_seconds: float = 30.0
    context_limit: int = 4096
    output_limit: int = 32
    temperature: float = 0.0
    cpu_threads: int = 4
    keep_alive: int = 0
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.allow_model_download, bool):
            raise ValueError("allow_model_download must be a bool")
        if self.through != "survival":
            raise ValueError("only the survival phase is executable")
        for name in ("pull_timeout_seconds", "request_timeout_seconds", "model_timeout_seconds",
                     "resource_wait_timeout_seconds", "resource_retry_seconds"):
            _positive(getattr(self, name), name)
        if self.context_limit != 4096 or self.output_limit != 32 or self.temperature != 0 or self.keep_alive != 0:
            raise ValueError("survival generation policy must remain 4096/32/temperature-zero/keep_alive-zero")
        if not isinstance(self.cpu_threads, int) or isinstance(self.cpu_threads, bool) or not 1 <= self.cpu_threads <= 8:
            raise ValueError("cpu_threads must be a bounded integer from one through eight")
        if self.max_concurrency != 1:
            raise ValueError("tournament max_concurrency must be exactly one")


@dataclass(frozen=True)
class ProvisioningEvidence:
    candidate_id: str
    model: str
    provider: str
    digest: str
    size_bytes: int
    parameter_size: str | None
    quantization: str | None
    capabilities: tuple[str, ...]
    provisioned_at: datetime
    already_installed: bool
    download_attempted: bool

    def __post_init__(self) -> None:
        for name in ("candidate_id", "model", "provider", "digest"):
            _text(getattr(self, name), name)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        for name in ("parameter_size", "quantization"):
            _text(getattr(self, name), name, optional=True)
        if not isinstance(self.capabilities, tuple) or any(not isinstance(item, str) or not item.strip() for item in self.capabilities):
            raise ValueError("capabilities must be a tuple of non-empty strings")
        _utc(self.provisioned_at, "provisioned_at")
        if not isinstance(self.already_installed, bool) or not isinstance(self.download_attempted, bool):
            raise ValueError("provisioning flags must be bool")
        if self.already_installed and self.download_attempted:
            raise ValueError("an already-installed model cannot also be downloaded")


def survival_attempt_id(tournament_id: str, candidate_id: str, attempt_number: int) -> str:
    """Return the stable identity for one admitted survival execution."""
    _text(tournament_id, "tournament_id")
    _text(candidate_id, "candidate_id")
    if not isinstance(attempt_number, int) or isinstance(attempt_number, bool) or attempt_number <= 0:
        raise ValueError("attempt_number must be a positive integer")
    return (
        f"{len(tournament_id)}:{tournament_id}|"
        f"{len(candidate_id)}:{candidate_id}|attempt:{attempt_number}"
    )


@dataclass(frozen=True)
class ResourceAdmissionEvidence:
    request_id: str
    decision: ResourceDecision
    reasons: tuple[str, ...]
    snapshot_timestamp: datetime | None = None
    ram_required_bytes: int | None = None
    ram_available_bytes: int | None = None
    ram_reserved_bytes: int | None = None
    ram_headroom_bytes: int | None = None
    disk_required_bytes: int | None = None
    disk_free_bytes: int | None = None
    disk_reserved_bytes: int | None = None
    disk_headroom_bytes: int | None = None
    selected_gpu_index: int | None = None
    gpu_total_vram_bytes: int | None = None
    gpu_free_vram_bytes: int | None = None
    gpu_reserved_vram_bytes: int | None = None
    gpu_headroom_bytes: int | None = None
    gpu_temperature_c: float | None = None

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        if self.decision is not ResourceDecision.ALLOW:
            raise ValueError("admitted attempt evidence requires an ALLOW decision")
        if not isinstance(self.reasons, tuple) or not self.reasons or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.reasons
        ):
            raise ValueError("admission reasons must be a non-empty tuple of non-empty strings")
        _utc(self.snapshot_timestamp, "snapshot_timestamp", optional=True)
        for name in (
            "ram_required_bytes", "ram_available_bytes", "ram_reserved_bytes",
            "disk_required_bytes", "disk_free_bytes", "disk_reserved_bytes",
            "selected_gpu_index", "gpu_total_vram_bytes", "gpu_free_vram_bytes",
            "gpu_reserved_vram_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer when present")
        for name in ("ram_headroom_bytes", "disk_headroom_bytes", "gpu_headroom_bytes"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{name} must be an integer when present")
        if self.gpu_temperature_c is not None and (
            not isinstance(self.gpu_temperature_c, (int, float))
            or isinstance(self.gpu_temperature_c, bool)
            or not math.isfinite(self.gpu_temperature_c)
        ):
            raise ValueError("gpu_temperature_c must be finite when present")

    @classmethod
    def capture(cls, assessment: object, request_id: str) -> "ResourceAdmissionEvidence":
        return cls(
            request_id=request_id,
            decision=getattr(assessment, "decision"),
            reasons=tuple(getattr(assessment, "reasons")),
            snapshot_timestamp=getattr(assessment, "snapshot_timestamp", None),
            **{
                name: getattr(assessment, name, None)
                for name in (
                    "ram_required_bytes", "ram_available_bytes", "ram_reserved_bytes",
                    "ram_headroom_bytes", "disk_required_bytes", "disk_free_bytes",
                    "disk_reserved_bytes", "disk_headroom_bytes", "selected_gpu_index",
                    "gpu_total_vram_bytes", "gpu_free_vram_bytes",
                    "gpu_reserved_vram_bytes", "gpu_headroom_bytes", "gpu_temperature_c",
                )
            },
        )


@dataclass(frozen=True)
class ActiveSurvivalAttempt:
    attempt_id: str
    tournament_id: str
    candidate_id: str
    attempt_number: int
    round_id: str
    task_id: str
    started_at: datetime
    resource_request: ResourceRequest
    admission: ResourceAdmissionEvidence

    def __post_init__(self) -> None:
        for name in ("attempt_id", "tournament_id", "candidate_id", "round_id", "task_id"):
            _text(getattr(self, name), name)
        if not isinstance(self.attempt_number, int) or isinstance(self.attempt_number, bool) or self.attempt_number <= 0:
            raise ValueError("attempt_number must be a positive integer")
        if self.attempt_id != survival_attempt_id(self.tournament_id, self.candidate_id, self.attempt_number):
            raise ValueError("attempt_id does not match its tournament/candidate/number identity")
        _utc(self.started_at, "started_at")
        if not isinstance(self.resource_request, ResourceRequest):
            raise ValueError("resource_request must be a ResourceRequest")
        if not isinstance(self.admission, ResourceAdmissionEvidence):
            raise ValueError("admission must be ResourceAdmissionEvidence")
        if self.admission.request_id != self.resource_request.id:
            raise ValueError("admission request identity mismatch")


@dataclass(frozen=True)
class SurvivalAttemptEvidence:
    attempt: ActiveSurvivalAttempt
    evidence_state: AttemptEvidenceState
    finished_at: datetime
    benchmark_outcome: BenchmarkOutcome | None
    benchmark_reasons: tuple[str, ...]
    benchmark_started_at: datetime | None = None
    benchmark_finished_at: datetime | None = None
    worker_pid: int | None = None
    worker_started_epoch: float | None = None
    owned_ollama_pid: int | None = None
    owned_ollama_started_epoch: float | None = None
    response_valid: bool | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    cleanup_verified: bool | None = None
    resource_recovery_verified: bool | None = None
    default_server_identity_preserved: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, ActiveSurvivalAttempt):
            raise ValueError("attempt must be ActiveSurvivalAttempt")
        if not isinstance(self.evidence_state, AttemptEvidenceState):
            raise ValueError("evidence_state must be AttemptEvidenceState")
        _utc(self.finished_at, "finished_at")
        if self.finished_at < self.attempt.started_at:
            raise ValueError("finished_at must not precede attempt start")
        if self.evidence_state is AttemptEvidenceState.FINALIZED and not isinstance(self.benchmark_outcome, BenchmarkOutcome):
            raise ValueError("finalized evidence requires a benchmark outcome")
        if self.benchmark_outcome is not None and not isinstance(self.benchmark_outcome, BenchmarkOutcome):
            raise ValueError("benchmark_outcome must be BenchmarkOutcome when present")
        if not isinstance(self.benchmark_reasons, tuple) or not self.benchmark_reasons or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.benchmark_reasons
        ):
            raise ValueError("benchmark_reasons must be a non-empty tuple of non-empty strings")
        _utc(self.benchmark_started_at, "benchmark_started_at", optional=True)
        _utc(self.benchmark_finished_at, "benchmark_finished_at", optional=True)
        if (
            self.benchmark_started_at is not None
            and self.benchmark_finished_at is not None
            and self.benchmark_finished_at < self.benchmark_started_at
        ):
            raise ValueError("benchmark timestamps are inconsistent")
        for prefix in ("worker", "owned_ollama"):
            pid = getattr(self, f"{prefix}_pid")
            started = getattr(self, f"{prefix}_started_epoch")
            if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0):
                raise ValueError(f"{prefix}_pid must be positive when present")
            if started is not None:
                if pid is None:
                    raise ValueError(f"{prefix}_started_epoch requires its PID")
                _positive(started, f"{prefix}_started_epoch")
        for name in (
            "response_valid", "cleanup_verified", "resource_recovery_verified",
            "default_server_identity_preserved",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be bool when present")
        object.__setattr__(self, "metrics", MappingProxyType({
            key: freeze_json(value) for key, value in dict(self.metrics).items()
        }))

    @property
    def attempt_id(self) -> str:
        return self.attempt.attempt_id


@dataclass(frozen=True)
class CandidateSurvivalRecord:
    candidate_id: str
    model: str
    provisioning_state: ProvisioningState = ProvisioningState.PENDING
    provisioning: ProvisioningEvidence | None = None
    outcome: SurvivalOutcome = SurvivalOutcome.PENDING
    attempt_count: int = 0
    reason: str | None = None
    worker_pid: int | None = None
    owned_ollama_pid: int | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    cleanup_verified: bool | None = None
    resource_recovery_verified: bool | None = None
    default_server_identity_preserved: bool | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        _text(self.model, "model")
        if not isinstance(self.provisioning_state, ProvisioningState) or not isinstance(self.outcome, SurvivalOutcome):
            raise ValueError("invalid candidate state")
        if self.provisioning is not None:
            if not isinstance(self.provisioning, ProvisioningEvidence):
                raise ValueError("provisioning must be ProvisioningEvidence")
            if (self.provisioning.candidate_id, self.provisioning.model) != (self.candidate_id, self.model):
                raise ValueError("provisioning identity mismatch")
        if self.provisioning_state is ProvisioningState.VERIFIED and self.provisioning is None:
            raise ValueError("verified provisioning requires evidence")
        if not isinstance(self.attempt_count, int) or isinstance(self.attempt_count, bool) or self.attempt_count < 0:
            raise ValueError("attempt_count must be a non-negative integer")
        _text(self.reason, "reason", optional=True)
        for name in ("worker_pid", "owned_ollama_pid"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer when present")
        for name in ("cleanup_verified", "resource_recovery_verified", "default_server_identity_preserved"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be bool when present")
        if self.outcome is SurvivalOutcome.PASSED and not all((self.cleanup_verified, self.resource_recovery_verified,
                                                               self.default_server_identity_preserved)):
            raise ValueError("survival PASS requires all cleanup and isolation gates")
        object.__setattr__(self, "metrics", MappingProxyType({key: freeze_json(value) for key, value in dict(self.metrics).items()}))


@dataclass(frozen=True)
class SurvivalCampaign:
    tournament_id: str
    manifest_digest: str
    state: SurvivalCampaignState
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime
    records: Mapping[str, CandidateSurvivalRecord]
    current_candidate_id: str | None = None
    current_round_id: str | None = None
    cancellation_requested: bool = False
    supervisor_pid: int | None = None
    supervisor_started_epoch: float | None = None
    supervisor_claim_token: str | None = None
    heartbeat_at: datetime | None = None
    next_retry_at: datetime | None = None
    last_event: str | None = None
    last_reason: str | None = None
    active_attempt: ActiveSurvivalAttempt | None = None

    def __post_init__(self) -> None:
        _text(self.tournament_id, "tournament_id")
        _text(self.manifest_digest, "manifest_digest")
        if not isinstance(self.state, SurvivalCampaignState):
            raise ValueError("state must be a SurvivalCampaignState")
        for name in ("created_at", "updated_at", "deadline_at"):
            _utc(getattr(self, name), name)
        if self.updated_at < self.created_at or self.deadline_at <= self.created_at:
            raise ValueError("campaign timestamps are inconsistent")
        copied = dict(self.records)
        if any(key != value.candidate_id or not isinstance(value, CandidateSurvivalRecord) for key, value in copied.items()):
            raise ValueError("records contain invalid candidate identities")
        if self.active_attempt is not None:
            if not isinstance(self.active_attempt, ActiveSurvivalAttempt):
                raise ValueError("active_attempt must be ActiveSurvivalAttempt")
            if (
                self.active_attempt.tournament_id != self.tournament_id
                or self.active_attempt.candidate_id not in copied
                or self.active_attempt.attempt_number
                != copied[self.active_attempt.candidate_id].attempt_count
                or copied[self.active_attempt.candidate_id].outcome is not SurvivalOutcome.RUNNING
                or self.current_candidate_id != self.active_attempt.candidate_id
                or self.current_round_id != self.active_attempt.round_id
            ):
                raise ValueError("active attempt does not match campaign identity or attempt count")
        for name in ("current_candidate_id", "current_round_id", "supervisor_claim_token", "last_event", "last_reason"):
            _text(getattr(self, name), name, optional=True)
        if not isinstance(self.cancellation_requested, bool):
            raise ValueError("cancellation_requested must be bool")
        if self.supervisor_pid is not None and (not isinstance(self.supervisor_pid, int) or isinstance(self.supervisor_pid, bool) or self.supervisor_pid <= 0):
            raise ValueError("supervisor_pid must be positive when present")
        if (self.supervisor_pid is None) != (self.supervisor_started_epoch is None):
            raise ValueError("supervisor identity must be complete")
        if self.supervisor_started_epoch is not None:
            _positive(self.supervisor_started_epoch, "supervisor_started_epoch")
        _utc(self.heartbeat_at, "heartbeat_at", optional=True)
        _utc(self.next_retry_at, "next_retry_at", optional=True)
        object.__setattr__(self, "records", MappingProxyType(copied))

    @property
    def progress(self) -> tuple[int, int]:
        completed = sum(record.outcome in TERMINAL_CANDIDATE_OUTCOMES for record in self.records.values())
        return completed, len(self.records)


def _provisioning_to_dict(item: ProvisioningEvidence) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id, "model": item.model, "provider": item.provider,
        "digest": item.digest, "size_bytes": item.size_bytes, "parameter_size": item.parameter_size,
        "quantization": item.quantization, "capabilities": list(item.capabilities),
        "provisioned_at": item.provisioned_at.isoformat(), "already_installed": item.already_installed,
        "download_attempted": item.download_attempted,
    }


def _request_to_dict(item: ResourceRequest) -> dict[str, Any]:
    return {
        "id": item.id,
        "workload_type": item.workload_type,
        "estimated_ram_bytes": item.estimated_ram_bytes,
        "estimated_disk_bytes": item.estimated_disk_bytes,
        "estimated_gpu_vram_bytes": item.estimated_gpu_vram_bytes,
        "gpu_index": item.gpu_index,
    }


def _request_from_dict(value: Mapping[str, Any]) -> ResourceRequest:
    return ResourceRequest(
        id=value["id"],
        workload_type=value["workload_type"],
        estimated_ram_bytes=value["estimated_ram_bytes"],
        estimated_disk_bytes=value["estimated_disk_bytes"],
        estimated_gpu_vram_bytes=value.get("estimated_gpu_vram_bytes", 0),
        gpu_index=value.get("gpu_index"),
    )


def _admission_to_dict(item: ResourceAdmissionEvidence) -> dict[str, Any]:
    return {
        "request_id": item.request_id,
        "decision": item.decision.value,
        "reasons": list(item.reasons),
        "snapshot_timestamp": None if item.snapshot_timestamp is None else item.snapshot_timestamp.isoformat(),
        **{
            name: getattr(item, name)
            for name in (
                "ram_required_bytes", "ram_available_bytes", "ram_reserved_bytes",
                "ram_headroom_bytes", "disk_required_bytes", "disk_free_bytes",
                "disk_reserved_bytes", "disk_headroom_bytes", "selected_gpu_index",
                "gpu_total_vram_bytes", "gpu_free_vram_bytes",
                "gpu_reserved_vram_bytes", "gpu_headroom_bytes", "gpu_temperature_c",
            )
        },
    }


def _admission_from_dict(value: Mapping[str, Any]) -> ResourceAdmissionEvidence:
    stamp = value.get("snapshot_timestamp")
    return ResourceAdmissionEvidence(
        request_id=value["request_id"],
        decision=ResourceDecision(value["decision"]),
        reasons=tuple(value["reasons"]),
        snapshot_timestamp=None if stamp is None else datetime.fromisoformat(stamp),
        **{
            name: value.get(name)
            for name in (
                "ram_required_bytes", "ram_available_bytes", "ram_reserved_bytes",
                "ram_headroom_bytes", "disk_required_bytes", "disk_free_bytes",
                "disk_reserved_bytes", "disk_headroom_bytes", "selected_gpu_index",
                "gpu_total_vram_bytes", "gpu_free_vram_bytes",
                "gpu_reserved_vram_bytes", "gpu_headroom_bytes", "gpu_temperature_c",
            )
        },
    )


def _active_attempt_to_dict(item: ActiveSurvivalAttempt) -> dict[str, Any]:
    return {
        "attempt_id": item.attempt_id,
        "tournament_id": item.tournament_id,
        "candidate_id": item.candidate_id,
        "attempt_number": item.attempt_number,
        "round_id": item.round_id,
        "task_id": item.task_id,
        "started_at": item.started_at.isoformat(),
        "resource_request": _request_to_dict(item.resource_request),
        "admission": _admission_to_dict(item.admission),
    }


def _active_attempt_from_dict(value: Mapping[str, Any]) -> ActiveSurvivalAttempt:
    return ActiveSurvivalAttempt(
        attempt_id=value["attempt_id"],
        tournament_id=value["tournament_id"],
        candidate_id=value["candidate_id"],
        attempt_number=value["attempt_number"],
        round_id=value["round_id"],
        task_id=value["task_id"],
        started_at=datetime.fromisoformat(value["started_at"]),
        resource_request=_request_from_dict(value["resource_request"]),
        admission=_admission_from_dict(value["admission"]),
    )


def attempt_evidence_to_dict(item: SurvivalAttemptEvidence) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "attempt": _active_attempt_to_dict(item.attempt),
        "evidence_state": item.evidence_state.value,
        "finished_at": item.finished_at.isoformat(),
        "benchmark_outcome": None if item.benchmark_outcome is None else item.benchmark_outcome.value,
        "benchmark_reasons": list(item.benchmark_reasons),
        "benchmark_started_at": None if item.benchmark_started_at is None else item.benchmark_started_at.isoformat(),
        "benchmark_finished_at": None if item.benchmark_finished_at is None else item.benchmark_finished_at.isoformat(),
        "worker_pid": item.worker_pid,
        "worker_started_epoch": item.worker_started_epoch,
        "owned_ollama_pid": item.owned_ollama_pid,
        "owned_ollama_started_epoch": item.owned_ollama_started_epoch,
        "response_valid": item.response_valid,
        "metrics": thaw_json(item.metrics),
        "cleanup_verified": item.cleanup_verified,
        "resource_recovery_verified": item.resource_recovery_verified,
        "default_server_identity_preserved": item.default_server_identity_preserved,
    }


def attempt_evidence_from_dict(value: Mapping[str, Any]) -> SurvivalAttemptEvidence:
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported attempt evidence schema: {value.get('schema_version')!r}")
    outcome = value.get("benchmark_outcome")
    benchmark_started = value.get("benchmark_started_at")
    benchmark_finished = value.get("benchmark_finished_at")
    return SurvivalAttemptEvidence(
        attempt=_active_attempt_from_dict(value["attempt"]),
        evidence_state=AttemptEvidenceState(value["evidence_state"]),
        finished_at=datetime.fromisoformat(value["finished_at"]),
        benchmark_outcome=None if outcome is None else BenchmarkOutcome(outcome),
        benchmark_reasons=tuple(value["benchmark_reasons"]),
        benchmark_started_at=None if benchmark_started is None else datetime.fromisoformat(benchmark_started),
        benchmark_finished_at=None if benchmark_finished is None else datetime.fromisoformat(benchmark_finished),
        worker_pid=value.get("worker_pid"),
        worker_started_epoch=value.get("worker_started_epoch"),
        owned_ollama_pid=value.get("owned_ollama_pid"),
        owned_ollama_started_epoch=value.get("owned_ollama_started_epoch"),
        response_valid=value.get("response_valid"),
        metrics=value.get("metrics", {}),
        cleanup_verified=value.get("cleanup_verified"),
        resource_recovery_verified=value.get("resource_recovery_verified"),
        default_server_identity_preserved=value.get("default_server_identity_preserved"),
    )


def campaign_to_dict(item: SurvivalCampaign) -> dict[str, Any]:
    return {
        "tournament_id": item.tournament_id, "manifest_digest": item.manifest_digest,
        "state": item.state.value, "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(), "deadline_at": item.deadline_at.isoformat(),
        "current_candidate_id": item.current_candidate_id, "current_round_id": item.current_round_id,
        "active_attempt": None if item.active_attempt is None else _active_attempt_to_dict(item.active_attempt),
        "cancellation_requested": item.cancellation_requested, "supervisor_pid": item.supervisor_pid,
        "supervisor_started_epoch": item.supervisor_started_epoch,
        "supervisor_claim_token": item.supervisor_claim_token,
        "heartbeat_at": None if item.heartbeat_at is None else item.heartbeat_at.isoformat(),
        "next_retry_at": None if item.next_retry_at is None else item.next_retry_at.isoformat(),
        "last_event": item.last_event, "last_reason": item.last_reason,
        "records": {key: {
            "candidate_id": value.candidate_id, "model": value.model,
            "provisioning_state": value.provisioning_state.value,
            "provisioning": None if value.provisioning is None else _provisioning_to_dict(value.provisioning),
            "outcome": value.outcome.value, "attempt_count": value.attempt_count,
            "reason": value.reason, "worker_pid": value.worker_pid,
            "owned_ollama_pid": value.owned_ollama_pid, "metrics": thaw_json(value.metrics),
            "cleanup_verified": value.cleanup_verified,
            "resource_recovery_verified": value.resource_recovery_verified,
            "default_server_identity_preserved": value.default_server_identity_preserved,
        } for key, value in sorted(item.records.items())},
    }


def campaign_from_dict(value: Mapping[str, Any]) -> SurvivalCampaign:
    try:
        records = {}
        for key, raw in value["records"].items():
            evidence_raw = raw.get("provisioning")
            evidence = None if evidence_raw is None else ProvisioningEvidence(
                candidate_id=evidence_raw["candidate_id"], model=evidence_raw["model"],
                provider=evidence_raw["provider"], digest=evidence_raw["digest"],
                size_bytes=evidence_raw["size_bytes"], parameter_size=evidence_raw.get("parameter_size"),
                quantization=evidence_raw.get("quantization"), capabilities=tuple(evidence_raw.get("capabilities", ())),
                provisioned_at=datetime.fromisoformat(evidence_raw["provisioned_at"]),
                already_installed=evidence_raw["already_installed"],
                download_attempted=evidence_raw["download_attempted"],
            )
            records[key] = CandidateSurvivalRecord(
                candidate_id=raw["candidate_id"], model=raw["model"],
                provisioning_state=ProvisioningState(raw["provisioning_state"]), provisioning=evidence,
                outcome=SurvivalOutcome(raw["outcome"]), attempt_count=raw.get("attempt_count", 0),
                reason=raw.get("reason"), worker_pid=raw.get("worker_pid"),
                owned_ollama_pid=raw.get("owned_ollama_pid"), metrics=raw.get("metrics", {}),
                cleanup_verified=raw.get("cleanup_verified"),
                resource_recovery_verified=raw.get("resource_recovery_verified"),
                default_server_identity_preserved=raw.get("default_server_identity_preserved"),
            )
        active_raw = value.get("active_attempt")
        return SurvivalCampaign(
            tournament_id=value["tournament_id"], manifest_digest=value["manifest_digest"],
            state=SurvivalCampaignState(value["state"]), created_at=datetime.fromisoformat(value["created_at"]),
            updated_at=datetime.fromisoformat(value["updated_at"]), deadline_at=datetime.fromisoformat(value["deadline_at"]),
            records=records,
            active_attempt=None if active_raw is None else _active_attempt_from_dict(active_raw),
            current_candidate_id=value.get("current_candidate_id"),
            current_round_id=value.get("current_round_id"), cancellation_requested=value.get("cancellation_requested", False),
            supervisor_pid=value.get("supervisor_pid"), supervisor_started_epoch=value.get("supervisor_started_epoch"),
            supervisor_claim_token=value.get("supervisor_claim_token"),
            heartbeat_at=None if value.get("heartbeat_at") is None else datetime.fromisoformat(value["heartbeat_at"]),
            next_retry_at=None if value.get("next_retry_at") is None else datetime.fromisoformat(value["next_retry_at"]),
            last_event=value.get("last_event"), last_reason=value.get("last_reason"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TournamentStoreError(f"invalid survival execution state: {exc}") from exc


def policy_to_dict(item: SurvivalExecutionPolicy) -> dict[str, Any]:
    return dict(item.__dict__)


class SurvivalStore:
    """Atomic survival state layered on immutable TournamentStore manifests.

    ``execution.json`` is authoritative for survival progress; the generic
    ``tournament.json`` remains the immutable base-manifest record.
    """

    def __init__(self, root: Path | str = Path("data/tournaments")) -> None:
        self.root = Path(root)
        self.tournaments = TournamentStore(self.root)

    def execution_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "execution.json"

    def policy_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "execution-policy.json"

    def supervisor_log_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "supervisor.log"

    def attempts_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "attempts"

    def _attempt_candidate_path(self, tournament_id: str, candidate_id: str) -> Path:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise TournamentStoreError("attempt candidate identity must be non-empty")
        candidate_key = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        return self.attempts_path(tournament_id) / candidate_key

    def attempt_path(self, tournament_id: str, candidate_id: str, attempt_number: int) -> Path:
        if not isinstance(attempt_number, int) or isinstance(attempt_number, bool) or attempt_number <= 0:
            raise TournamentStoreError("attempt number must be positive")
        return self._attempt_candidate_path(tournament_id, candidate_id) / f"{attempt_number:06d}.json"

    def attempt_evidence_available(self, tournament_id: str) -> bool:
        """Distinguish new campaigns from historical campaigns without attempt evidence."""
        return self.attempts_path(tournament_id).is_dir()

    def attempts_for(self, tournament_id: str, candidate_id: str) -> tuple[SurvivalAttemptEvidence, ...]:
        """Return immutable admitted-attempt evidence in attempt-number order."""
        campaign = self.load(tournament_id)
        if candidate_id not in campaign.records:
            raise TournamentStoreError("attempt candidate is not in the persisted campaign")
        directory = self._attempt_candidate_path(tournament_id, candidate_id)
        if not directory.is_dir():
            return ()
        attempts = []
        for path in sorted(directory.glob("*.json")):
            try:
                evidence = attempt_evidence_from_dict(
                    TournamentStore._read_json(path, "survival attempt evidence")
                )
            except (TypeError, ValueError) as exc:
                raise TournamentStoreError(f"invalid survival attempt evidence: {exc}") from exc
            if (
                evidence.attempt.tournament_id != tournament_id
                or evidence.attempt.candidate_id != candidate_id
                or path != self.attempt_path(tournament_id, candidate_id, evidence.attempt.attempt_number)
            ):
                raise TournamentStoreError("attempt evidence path identity mismatch")
            attempts.append(evidence)
        attempts.sort(key=lambda item: item.attempt.attempt_number)
        numbers = [item.attempt.attempt_number for item in attempts]
        if numbers != sorted(set(numbers)):
            raise TournamentStoreError("attempt evidence identities are not unique and ordered")
        return tuple(attempts)

    def attempts_by_candidate(self, tournament_id: str) -> Mapping[str, tuple[SurvivalAttemptEvidence, ...]]:
        campaign = self.load(tournament_id)
        return MappingProxyType({
            candidate_id: self.attempts_for(tournament_id, candidate_id)
            for candidate_id in campaign.records
        })

    def finalize_attempt(self, evidence: SurvivalAttemptEvidence) -> SurvivalAttemptEvidence:
        """Atomically publish one immutable terminal record; identical replay is idempotent."""
        if not isinstance(evidence, SurvivalAttemptEvidence):
            raise TournamentStoreError("attempt evidence must be SurvivalAttemptEvidence")
        campaign = self.load(evidence.attempt.tournament_id)
        record = campaign.records.get(evidence.attempt.candidate_id)
        if record is None or evidence.attempt.attempt_number > record.attempt_count:
            raise TournamentStoreError("attempt evidence does not match admitted campaign state")
        path = self.attempt_path(
            evidence.attempt.tournament_id,
            evidence.attempt.candidate_id,
            evidence.attempt.attempt_number,
        )
        payload = attempt_evidence_to_dict(evidence)
        if path.is_file():
            existing = attempt_evidence_from_dict(
                TournamentStore._read_json(path, "survival attempt evidence")
            )
            if existing != evidence:
                raise TournamentStoreError(f"immutable attempt evidence conflict: {evidence.attempt_id}")
            return existing
        if campaign.active_attempt != evidence.attempt:
            raise TournamentStoreError("new attempt evidence does not match the persisted active attempt")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = attempt_evidence_from_dict(
                    TournamentStore._read_json(path, "survival attempt evidence")
                )
                if existing != evidence:
                    raise TournamentStoreError(f"immutable attempt evidence conflict: {evidence.attempt_id}")
                return existing
        finally:
            temporary.unlink(missing_ok=True)
        return evidence

    def recover_active_attempt(
        self,
        campaign: SurvivalCampaign,
        *,
        finished_at: datetime,
        ownership_state: Mapping[str, object] | None = None,
        reason: str = "admitted execution was interrupted before terminal evidence was recorded",
    ) -> SurvivalCampaign:
        """Finalize or reconcile an interrupted mutable attempt without duplicating history."""
        active = campaign.active_attempt
        if active is None:
            return campaign
        existing = next((
            item for item in self.attempts_for(campaign.tournament_id, active.candidate_id)
            if item.attempt_id == active.attempt_id
        ), None)
        if existing is None:
            state = dict(ownership_state or {})

            def identity(kind: str) -> tuple[int | None, float | None]:
                pid = state.get(f"{kind}_pid")
                started = state.get(f"{kind}_started_epoch")
                if (
                    isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
                    and isinstance(started, (int, float)) and not isinstance(started, bool)
                    and math.isfinite(started) and started > 0
                ):
                    return pid, float(started)
                return None, None

            worker_pid, worker_started = identity("worker")
            server_pid, server_started = identity("server")
            self.finalize_attempt(SurvivalAttemptEvidence(
                attempt=active,
                evidence_state=AttemptEvidenceState.INTERRUPTED,
                finished_at=finished_at,
                benchmark_outcome=None,
                benchmark_reasons=(reason,),
                worker_pid=worker_pid,
                worker_started_epoch=worker_started,
                owned_ollama_pid=server_pid,
                owned_ollama_started_epoch=server_started,
            ))
        records = dict(campaign.records)
        record = records[active.candidate_id]
        if record.outcome is SurvivalOutcome.RUNNING:
            records[active.candidate_id] = replace(
                record,
                outcome=SurvivalOutcome.INTERRUPTED,
                reason=reason,
            )
        return replace(campaign, records=records, active_attempt=None)

    def create(self, campaign: SurvivalCampaign, manifest: TournamentManifest,
               policy: SurvivalExecutionPolicy) -> SurvivalCampaign:
        if campaign.manifest_digest != manifest.digest:
            raise TournamentStoreError("campaign manifest digest mismatch")
        base = PersistedTournament(
            SCHEMA_VERSION, campaign.tournament_id, manifest.digest, "manifest.json",
            TournamentState.CREATED, campaign.created_at, campaign.created_at,
            progress_total=len(manifest.candidates),
            candidate_states={candidate.candidate_id: CandidateState.AVAILABLE for candidate in manifest.candidates},
        )
        self.tournaments.create(base, manifest)
        TournamentStore._atomic_json(self.policy_path(campaign.tournament_id), policy_to_dict(policy))
        TournamentStore._atomic_json(self.execution_path(campaign.tournament_id), campaign_to_dict(campaign))
        self.attempts_path(campaign.tournament_id).mkdir(exist_ok=False)
        self.append_event(campaign.tournament_id, "tournament created", campaign.created_at)
        return campaign

    def load(self, tournament_id: str) -> SurvivalCampaign:
        raw = TournamentStore._read_json(self.execution_path(tournament_id), "survival execution state")
        campaign = campaign_from_dict(raw)
        if campaign.tournament_id != tournament_id:
            raise TournamentStoreError("execution state tournament identity mismatch")
        self.tournaments.load_manifest(tournament_id).verify_digest(campaign.manifest_digest)
        return campaign

    def load_policy(self, tournament_id: str) -> SurvivalExecutionPolicy:
        raw = TournamentStore._read_json(self.policy_path(tournament_id), "execution policy")
        try:
            return SurvivalExecutionPolicy(**raw)
        except (TypeError, ValueError) as exc:
            raise TournamentStoreError(f"invalid execution policy: {exc}") from exc

    def save(self, campaign: SurvivalCampaign, *, allow_resume: bool = False) -> SurvivalCampaign:
        previous = self.load(campaign.tournament_id)
        if campaign.manifest_digest != previous.manifest_digest or campaign.updated_at < previous.updated_at:
            raise TournamentStoreError("survival state identity or timestamp regressed")

        if campaign.state is not previous.state:
            explicit_resume = (
                allow_resume
                and previous.state is SurvivalCampaignState.INTERRUPTED
                and campaign.state is SurvivalCampaignState.CREATED
            )
            if not explicit_resume and campaign.state not in SURVIVAL_STATE_TRANSITIONS[previous.state]:
                raise TournamentStoreError(
                    f"illegal survival state transition: "
                    f"{previous.state.value} -> {campaign.state.value}"
                )

        if (
            previous.state in {
                SurvivalCampaignState.COMPLETED,
                SurvivalCampaignState.CANCELLED,
                SurvivalCampaignState.FAILED,
            }
            and campaign != previous
        ):
            raise TournamentStoreError("terminal survival state is immutable")
        if previous.records.keys() != campaign.records.keys():
            raise TournamentStoreError("candidate record set is immutable")
        for candidate_id, old in previous.records.items():
            new = campaign.records[candidate_id]
            if old.provisioning is not None and new.provisioning != old.provisioning:
                raise TournamentStoreError("verified provisioning evidence is immutable")
            if old.outcome in TERMINAL_CANDIDATE_OUTCOMES and new != old:
                raise TournamentStoreError("terminal candidate evidence is immutable")
        TournamentStore._atomic_json(self.execution_path(campaign.tournament_id), campaign_to_dict(campaign))
        return campaign

    def append_event(self, tournament_id: str, event: str, timestamp: datetime,
                     reason: str | None = None, candidate_id: str | None = None) -> None:
        self.tournaments.append_event(tournament_id, event, timestamp, reason)
        # The base event writer deliberately has a small schema. Candidate context
        # is encoded in the reason without rewriting prior append-only evidence.
        if candidate_id is not None and reason is None:
            del candidate_id

    def list(self) -> tuple[SurvivalCampaign, ...]:
        values = []
        if self.root.is_dir():
            for directory in self.root.iterdir():
                if directory.is_dir() and (directory / "execution.json").is_file():
                    values.append(self.load(directory.name))
        return tuple(sorted(values, key=lambda item: (item.created_at, item.tournament_id), reverse=True))

    def latest(self) -> SurvivalCampaign | None:
        values = self.list()
        return values[0] if values else None

    def active(self) -> SurvivalCampaign | None:
        return next((item for item in self.list() if item.state in ACTIVE_SURVIVAL_STATES), None)

