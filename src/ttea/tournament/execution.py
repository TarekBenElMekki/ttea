"""Persistent contracts for the controlled survival tournament."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

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


def campaign_to_dict(item: SurvivalCampaign) -> dict[str, Any]:
    return {
        "tournament_id": item.tournament_id, "manifest_digest": item.manifest_digest,
        "state": item.state.value, "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(), "deadline_at": item.deadline_at.isoformat(),
        "current_candidate_id": item.current_candidate_id, "current_round_id": item.current_round_id,
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
        return SurvivalCampaign(
            tournament_id=value["tournament_id"], manifest_digest=value["manifest_digest"],
            state=SurvivalCampaignState(value["state"]), created_at=datetime.fromisoformat(value["created_at"]),
            updated_at=datetime.fromisoformat(value["updated_at"]), deadline_at=datetime.fromisoformat(value["deadline_at"]),
            records=records, current_candidate_id=value.get("current_candidate_id"),
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
    """Atomic survival state layered on immutable TournamentStore manifests."""

    def __init__(self, root: Path | str = Path("data/tournaments")) -> None:
        self.root = Path(root)
        self.tournaments = TournamentStore(self.root)

    def execution_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "execution.json"

    def policy_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "execution-policy.json"

    def supervisor_log_path(self, tournament_id: str) -> Path:
        return self.tournaments.tournament_path(tournament_id) / "supervisor.log"

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

