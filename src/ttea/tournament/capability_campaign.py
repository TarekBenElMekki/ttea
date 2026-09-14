"""Persistent orchestration, provenance preflight, and reports for Capability Round 2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import uuid4

import psutil

from ttea.benchmarking import BenchmarkStateStore
from ttea.resources import ResourceManager
from ttea.telemetry import HardwareTelemetryService

from .capability import (
    ROUND2_CANDIDATES,
    ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST,
    ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID,
    CandidateCapabilitySummary,
    CapabilityStoreError,
    CapabilityWorkId,
    CapabilityWorkStore,
    capability_work_order,
    load_round2_manifest,
    rank_candidates,
    summarize_candidate,
    terminal_state,
)
from .capability_runtime import CapabilityCampaignResult, CapabilityRuntimeEngine
from .execution import (
    ProvisioningState,
    SurvivalCampaignState,
    SurvivalExecutionPolicy,
    SurvivalOutcome,
    SurvivalStore,
)
from .ollama_provider import (
    OllamaTournamentProvider,
    ProvisioningDeferred,
    ProvisioningRejected,
    ProvisioningRuntimeDelayed,
)


CAPABILITY_CAMPAIGN_SCHEMA_VERSION = 1
CAPABILITY_REPORT_SCHEMA_VERSION = 1
CAPABILITY_PHASE = "CAPABILITY_ROUND_2"
DEFAULT_BENCHMARK_STATE = Path("data/runtime/benchmark.json")


class CapabilityCampaignState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED_ELIGIBLE = "COMPLETED_ELIGIBLE"
    COMPLETED_NO_ELIGIBLE_MODEL = "COMPLETED_NO_ELIGIBLE_MODEL"
    COMPLETED_INCOMPLETE = "COMPLETED_INCOMPLETE"
    BLOCKED_INTEGRITY = "BLOCKED_INTEGRITY"
    FAILED = "FAILED"


TERMINAL_CAPABILITY_CAMPAIGN_STATES = frozenset((
    CapabilityCampaignState.COMPLETED_ELIGIBLE,
    CapabilityCampaignState.COMPLETED_NO_ELIGIBLE_MODEL,
    CapabilityCampaignState.COMPLETED_INCOMPLETE,
    CapabilityCampaignState.BLOCKED_INTEGRITY,
    CapabilityCampaignState.FAILED,
))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class CapabilityCampaign:
    tournament_id: str
    source_survival_tournament_id: str
    source_survival_manifest_digest: str
    candidate_manifest_digest: str
    state: CapabilityCampaignState
    created_at: datetime
    updated_at: datetime
    owner_pid: int | None = None
    owner_started_epoch: float | None = None
    last_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "tournament_id",
            "source_survival_tournament_id",
            "source_survival_manifest_digest",
            "candidate_manifest_digest",
        ):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.state, CapabilityCampaignState):
            raise ValueError("state must be CapabilityCampaignState")
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() != timezone.utc.utcoffset(value)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if (self.owner_pid is None) != (self.owner_started_epoch is None):
            raise ValueError("campaign owner identity must be complete")
        if self.owner_pid is not None and (
            not isinstance(self.owner_pid, int)
            or isinstance(self.owner_pid, bool)
            or self.owner_pid <= 0
        ):
            raise ValueError("owner_pid must be positive")
        if self.owner_started_epoch is not None and (
            not isinstance(self.owner_started_epoch, (int, float))
            or isinstance(self.owner_started_epoch, bool)
            or self.owner_started_epoch <= 0
        ):
            raise ValueError("owner_started_epoch must be positive")
        if self.last_reason is not None:
            _non_empty(self.last_reason, "last_reason")


def _campaign_to_dict(campaign: CapabilityCampaign) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_CAMPAIGN_SCHEMA_VERSION,
        "tournament_id": campaign.tournament_id,
        "source_survival_tournament_id": campaign.source_survival_tournament_id,
        "source_survival_manifest_digest": campaign.source_survival_manifest_digest,
        "candidate_manifest_digest": campaign.candidate_manifest_digest,
        "state": campaign.state.value,
        "created_at": campaign.created_at.isoformat(),
        "updated_at": campaign.updated_at.isoformat(),
        "owner_pid": campaign.owner_pid,
        "owner_started_epoch": campaign.owner_started_epoch,
        "last_reason": campaign.last_reason,
    }


def _campaign_from_dict(value: Mapping[str, Any]) -> CapabilityCampaign:
    if value.get("schema_version") != CAPABILITY_CAMPAIGN_SCHEMA_VERSION:
        raise CapabilityStoreError("unsupported Capability campaign schema")
    try:
        return CapabilityCampaign(
            value["tournament_id"],
            value["source_survival_tournament_id"],
            value["source_survival_manifest_digest"],
            value["candidate_manifest_digest"],
            CapabilityCampaignState(value["state"]),
            datetime.fromisoformat(value["created_at"]),
            datetime.fromisoformat(value["updated_at"]),
            value.get("owner_pid"),
            value.get("owner_started_epoch"),
            value.get("last_reason"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CapabilityStoreError(f"invalid Capability campaign state: {exc}") from exc


class CapabilityCampaignStore:
    """Atomic campaign metadata layered over immutable per-work evidence."""

    def __init__(self, root: Path | str = Path("data/tournaments")) -> None:
        self.root = Path(root)
        self.work = CapabilityWorkStore(self.root)

    def campaign_path(self, tournament_id: str) -> Path:
        _non_empty(tournament_id, "tournament_id")
        return self.root / tournament_id / "capability-campaign.json"

    def report_json_path(self, tournament_id: str) -> Path:
        return self.root / tournament_id / "capability-report.json"

    def report_text_path(self, tournament_id: str) -> Path:
        return self.root / tournament_id / "capability-report.txt"

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, campaign: CapabilityCampaign) -> CapabilityCampaign:
        path = self.campaign_path(campaign.tournament_id)
        if path.exists():
            existing = self.load(campaign.tournament_id)
            if existing != campaign:
                raise CapabilityStoreError("Capability campaign identity conflict")
            return existing
        self._atomic_json(path, _campaign_to_dict(campaign))
        return campaign

    def load(self, tournament_id: str) -> CapabilityCampaign:
        path = self.campaign_path(tournament_id)
        if not path.is_file():
            raise CapabilityStoreError("unknown Capability campaign")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CapabilityStoreError(f"cannot load Capability campaign: {exc}") from exc
        campaign = _campaign_from_dict(raw)
        if campaign.tournament_id != tournament_id:
            raise CapabilityStoreError("Capability campaign path identity mismatch")
        return campaign

    def save(self, campaign: CapabilityCampaign) -> CapabilityCampaign:
        previous = self.load(campaign.tournament_id)
        if (
            campaign.source_survival_tournament_id != previous.source_survival_tournament_id
            or campaign.source_survival_manifest_digest != previous.source_survival_manifest_digest
            or campaign.candidate_manifest_digest != previous.candidate_manifest_digest
            or campaign.created_at != previous.created_at
        ):
            raise CapabilityStoreError("Capability campaign provenance is immutable")
        if campaign.updated_at < previous.updated_at:
            raise CapabilityStoreError("Capability campaign timestamp regressed")
        if previous.state in TERMINAL_CAPABILITY_CAMPAIGN_STATES and campaign != previous:
            raise CapabilityStoreError("terminal Capability campaign is immutable")
        self._atomic_json(self.campaign_path(campaign.tournament_id), _campaign_to_dict(campaign))
        return campaign

    def list(self) -> tuple[CapabilityCampaign, ...]:
        campaigns = []
        if self.root.is_dir():
            for directory in self.root.iterdir():
                path = directory / "capability-campaign.json"
                if path.is_file():
                    campaigns.append(self.load(directory.name))
        return tuple(sorted(
            campaigns,
            key=lambda item: (item.created_at, item.tournament_id),
            reverse=True,
        ))

    def latest(self) -> CapabilityCampaign | None:
        values = self.list()
        return values[0] if values else None

    def progress(self, tournament_id: str) -> tuple[int, int, CapabilityWorkId | None]:
        completed = 0
        current = None
        order = capability_work_order(tournament_id)
        for work in order:
            path = self.root / tournament_id / "capability" / work.key / "work.json"
            if not path.is_file():
                if current is None:
                    current = work
                continue
            record = self.work.load(work)
            if record.terminal_status is not None:
                completed += 1
            elif current is None:
                current = work
        return completed, len(order), current

    def owner_alive(self, campaign: CapabilityCampaign) -> bool:
        if campaign.owner_pid is None:
            return False
        return BenchmarkStateStore.process_matches(
            campaign.owner_pid,
            campaign.owner_started_epoch,
        )


def _summary_to_dict(summary: CandidateCapabilitySummary) -> dict[str, Any]:
    return {
        "candidate_id": summary.candidate_id,
        "status": summary.status,
        "planning": summary.planning,
        "replanning": summary.replanning,
        "final_score": summary.final_score,
        "response_bytes": summary.response_bytes,
        "plan_steps": summary.plan_steps,
        "failures": list(summary.failures),
    }


def _derive_summaries(
    tournament_id: str,
    store: CapabilityCampaignStore,
) -> tuple[CandidateCapabilitySummary, ...]:
    by_candidate = []
    fixture_ids = {work.fixture_id for work in capability_work_order(tournament_id)}
    for candidate_id in ROUND2_CANDIDATES:
        results = {}
        incomplete = False
        for work in capability_work_order(tournament_id):
            if work.candidate_id != candidate_id:
                continue
            record_path = store.root / tournament_id / "capability" / work.key / "work.json"
            if not record_path.is_file():
                incomplete = True
                continue
            record = store.work.load(work)
            if record.evaluator_result_digest:
                results[work.fixture_id] = store.work.evaluator_result(work)
            else:
                incomplete = True
        if set(results) - fixture_ids:
            raise CapabilityStoreError("Capability summary references unknown fixture")
        by_candidate.append(
            summarize_candidate(candidate_id, results, incomplete=incomplete)
        )
    return tuple(by_candidate)


def capability_report_payload(
    campaign: CapabilityCampaign,
    store: CapabilityCampaignStore,
) -> dict[str, Any]:
    summaries = _derive_summaries(campaign.tournament_id, store)
    ranking = rank_candidates(summaries)
    completed, total, current = store.progress(campaign.tournament_id)
    work_refs = []
    for work in capability_work_order(campaign.tournament_id):
        path = store.root / campaign.tournament_id / "capability" / work.key / "work.json"
        if not path.is_file():
            work_refs.append({
                "work_id": work.to_dict(),
                "phase": "PENDING",
                "terminal_status": None,
                "raw_response_digest": None,
                "technical_evidence_refs": [],
                "evaluator_result_digest": None,
            })
            continue
        record = store.work.load(work)
        work_refs.append({
            "work_id": work.to_dict(),
            "phase": record.phase,
            "terminal_status": record.terminal_status,
            "raw_response_digest": record.raw_response_digest,
            "technical_evidence_refs": list(record.technical_evidence_refs),
            "evaluator_result_digest": record.evaluator_result_digest,
        })
    return {
        "schema_version": CAPABILITY_REPORT_SCHEMA_VERSION,
        "tournament_id": campaign.tournament_id,
        "phase": CAPABILITY_PHASE,
        "state": campaign.state.value,
        "source_survival": {
            "tournament_id": campaign.source_survival_tournament_id,
            "manifest_digest": campaign.source_survival_manifest_digest,
        },
        "candidate_manifest_digest": campaign.candidate_manifest_digest,
        "created_at": campaign.created_at.isoformat(),
        "updated_at": campaign.updated_at.isoformat(),
        "last_reason": campaign.last_reason,
        "progress": {
            "completed": completed,
            "total": total,
            "current_work": None if current is None else current.to_dict(),
        },
        "candidate_summaries": [_summary_to_dict(item) for item in summaries],
        "ranking": [item.candidate_id for item in ranking],
        "work_evidence": work_refs,
    }


def capability_report_text(
    campaign: CapabilityCampaign,
    store: CapabilityCampaignStore,
) -> str:
    payload = capability_report_payload(campaign, store)
    progress = payload["progress"]
    lines = [
        "T/TEA Capability Round 2",
        "========================",
        f"Tournament: {campaign.tournament_id}",
        f"State: {campaign.state.value}",
        f"Source Survival: {campaign.source_survival_tournament_id}",
        f"Source manifest: {campaign.source_survival_manifest_digest}",
        f"Candidate manifest: {campaign.candidate_manifest_digest}",
        f"Progress: {progress['completed']}/{progress['total']}",
        f"Last reason: {campaign.last_reason or '(none)'}",
        "",
        "Candidates:",
    ]
    for item in payload["candidate_summaries"]:
        score = "n/a" if item["final_score"] is None else f"{item['final_score']:.2f}"
        lines.append(
            f"- {item['candidate_id']}: {item['status']} | score={score} "
            f"| bytes={item['response_bytes']} | steps={item['plan_steps']}"
        )
    lines.append("")
    lines.append(
        "Ranking: " + (
            " > ".join(payload["ranking"])
            if payload["ranking"]
            else "(no eligible ranking)"
        )
    )
    return "\n".join(lines) + "\n"


def write_capability_report(
    campaign: CapabilityCampaign,
    store: CapabilityCampaignStore,
) -> None:
    payload = capability_report_payload(campaign, store)
    store._atomic_json(store.report_json_path(campaign.tournament_id), payload)
    path = store.report_text_path(campaign.tournament_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(capability_report_text(campaign, store))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CapabilityCampaignManager:
    """Foreground Capability orchestration with durable resume and exact source provenance."""

    def __init__(
        self,
        root: Path | str = Path("data/tournaments"),
        *,
        benchmark_state_path: Path = DEFAULT_BENCHMARK_STATE,
        process_api=psutil.Process,
        resource_manager_factory: Callable[[], ResourceManager] | None = None,
        provider_factory=OllamaTournamentProvider,
        engine_factory=CapabilityRuntimeEngine,
    ) -> None:
        self.store = CapabilityCampaignStore(root)
        self.survival_store = SurvivalStore(root)
        self.benchmark_state_path = Path(benchmark_state_path)
        self.process_api = process_api
        self.resource_manager_factory = (
            resource_manager_factory
            or (lambda: ResourceManager(HardwareTelemetryService(Path.cwd())))
        )
        self.provider_factory = provider_factory
        self.engine_factory = engine_factory

    def _source_campaign(self):
        source = self.survival_store.load(ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID)
        if source.manifest_digest != ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST:
            raise CapabilityStoreError("authoritative Survival campaign digest mismatch")
        if source.state is not SurvivalCampaignState.COMPLETED:
            raise CapabilityStoreError("authoritative Survival campaign is not completed")
        for candidate_id in ROUND2_CANDIDATES:
            record = source.records.get(candidate_id)
            if record is None:
                raise CapabilityStoreError(f"Survival source is missing survivor: {candidate_id}")
            if record.outcome is not SurvivalOutcome.PASSED:
                raise CapabilityStoreError(f"Survival source candidate did not pass: {candidate_id}")
            if record.provisioning_state is not ProvisioningState.VERIFIED or record.provisioning is None:
                raise CapabilityStoreError(f"Survival source has no verified provisioning evidence: {candidate_id}")
        return source

    def _new_campaign(self) -> CapabilityCampaign:
        manifest = load_round2_manifest(self.store.root)
        source = self._source_campaign()
        now = _utc_now()
        tournament_id = f"capability-{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
        identity = self.process_api(os.getpid())
        return CapabilityCampaign(
            tournament_id,
            source.tournament_id,
            source.manifest_digest,
            manifest.digest,
            CapabilityCampaignState.CREATED,
            now,
            now,
            os.getpid(),
            identity.create_time(),
            "Capability Round 2 campaign created from authoritative Survival evidence",
        )

    def _status_reconcile(
        self,
        campaign: CapabilityCampaign,
    ) -> tuple[CapabilityCampaign, bool]:
        alive = self.store.owner_alive(campaign)
        if campaign.state is CapabilityCampaignState.RUNNING and not alive:
            campaign = self.store.save(replace(
                campaign,
                state=CapabilityCampaignState.INTERRUPTED,
                updated_at=_utc_now(),
                owner_pid=None,
                owner_started_epoch=None,
                last_reason="Capability campaign owner identity is stale",
            ))
            write_capability_report(campaign, self.store)
        return campaign, alive

    def status(
        self,
        tournament_id: str | None = None,
    ) -> tuple[CapabilityCampaign | None, bool]:
        campaign = (
            self.store.load(tournament_id)
            if tournament_id
            else self.store.latest()
        )
        if campaign is None:
            return None, False
        return self._status_reconcile(campaign)

    def _preflight_source_models(
        self,
        provider: OllamaTournamentProvider,
        source,
        manifest,
    ) -> None:
        for candidate_id in ROUND2_CANDIDATES:
            candidate = manifest.candidate(candidate_id)
            prior = source.records[candidate_id].provisioning
            provider.provision(candidate, prior=prior)

    def _run_claimed(self, campaign: CapabilityCampaign) -> CapabilityCampaign:
        source = self._source_campaign()
        manifest = load_round2_manifest(self.store.root)
        manifest.verify_digest(campaign.candidate_manifest_digest)
        manager = self.resource_manager_factory()
        policy = SurvivalExecutionPolicy(allow_model_download=False)
        provider = self.provider_factory(
            manifest,
            manager,
            policy,
            state_path=self.benchmark_state_path,
        )
        work_store = self.store.work
        running = self.store.save(replace(
            campaign,
            state=CapabilityCampaignState.RUNNING,
            updated_at=_utc_now(),
            last_reason="verifying exact survived model identities before Capability execution",
        ))
        try:
            self._preflight_source_models(provider, source, manifest)
            running = self.store.save(replace(
                running,
                updated_at=_utc_now(),
                last_reason="exact survived model identities verified; Capability execution running",
            ))
            result: CapabilityCampaignResult = self.engine_factory(
                running.tournament_id,
                work_store,
                manifest,
                provider,
                manager,
            ).run()
            state = CapabilityCampaignState(result.terminal)
            final = self.store.save(replace(
                running,
                state=state,
                updated_at=_utc_now(),
                owner_pid=None,
                owner_started_epoch=None,
                last_reason=(
                    result.integrity_reason
                    or f"Capability runtime finished with {state.value}"
                ),
            ))
            write_capability_report(final, self.store)
            return final
        except KeyboardInterrupt:
            interrupted = self.store.save(replace(
                self.store.load(running.tournament_id),
                state=CapabilityCampaignState.INTERRUPTED,
                updated_at=_utc_now(),
                owner_pid=None,
                owner_started_epoch=None,
                last_reason="Capability campaign interrupted by operator",
            ))
            write_capability_report(interrupted, self.store)
            return interrupted
        except ProvisioningRejected as exc:
            blocked = self.store.save(replace(
                self.store.load(running.tournament_id),
                state=CapabilityCampaignState.BLOCKED_INTEGRITY,
                updated_at=_utc_now(),
                owner_pid=None,
                owner_started_epoch=None,
                last_reason=f"survived model identity verification failed: {exc}",
            ))
            write_capability_report(blocked, self.store)
            return blocked
        except (ProvisioningDeferred, ProvisioningRuntimeDelayed) as exc:
            interrupted = self.store.save(replace(
                self.store.load(running.tournament_id),
                state=CapabilityCampaignState.INTERRUPTED,
                updated_at=_utc_now(),
                owner_pid=None,
                owner_started_epoch=None,
                last_reason=f"local runtime preflight deferred: {exc}",
            ))
            write_capability_report(interrupted, self.store)
            return interrupted
        except BaseException as exc:
            failed = self.store.save(replace(
                self.store.load(running.tournament_id),
                state=CapabilityCampaignState.FAILED,
                updated_at=_utc_now(),
                owner_pid=None,
                owner_started_epoch=None,
                last_reason=f"{type(exc).__name__}: {exc}",
            ))
            write_capability_report(failed, self.store)
            return failed

    def start(self) -> CapabilityCampaign:
        latest = self.store.latest()
        if latest is not None and latest.state is CapabilityCampaignState.RUNNING:
            latest, alive = self._status_reconcile(latest)
            if alive:
                raise RuntimeError(f"active Capability campaign {latest.tournament_id} is already running")
        campaign = self._new_campaign()
        self.store.create(campaign)
        return self._run_claimed(campaign)

    def resume(self, tournament_id: str | None = None) -> CapabilityCampaign:
        campaign, alive = self.status(tournament_id)
        if campaign is None:
            raise RuntimeError("no Capability campaign exists")
        if alive:
            raise RuntimeError("active Capability campaign cannot be resumed")
        if campaign.state is not CapabilityCampaignState.INTERRUPTED:
            raise RuntimeError(f"Capability campaign in state {campaign.state.value} is not resumable")
        identity = self.process_api(os.getpid())
        resumed = self.store.save(replace(
            campaign,
            state=CapabilityCampaignState.CREATED,
            updated_at=_utc_now(),
            owner_pid=os.getpid(),
            owner_started_epoch=identity.create_time(),
            last_reason="Capability campaign resumed",
        ))
        return self._run_claimed(resumed)