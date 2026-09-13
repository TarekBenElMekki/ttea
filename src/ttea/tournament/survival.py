"""Sequential controlled provisioning and first-round survival orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import json
import time
from typing import Any, Callable

from ttea.benchmarking import BenchmarkOutcome
from ttea.resources import ResourceDecision, ResourceRequest

from .execution import (
    CandidateSurvivalRecord, ProvisioningState, SurvivalCampaign,
    SurvivalCampaignState, SurvivalExecutionPolicy, SurvivalOutcome,
    SurvivalStore, TERMINAL_CANDIDATE_OUTCOMES, campaign_to_dict, utc_now,
)
from .manifest import TournamentManifest
from .models import EliminationReason
from .ollama_provider import (
    OllamaTournamentProvider, ProvisioningDeferred, ProvisioningRejected,
    ProvisioningResourceDelayed, ProvisioningResourceDenied,
)
from .store import TournamentStore


def _event_reason(candidate_id: str, reason: str | None = None) -> str:
    return f"{candidate_id}: {reason}" if reason else candidate_id


def _status_summary(campaign: SurvivalCampaign) -> dict[str, int]:
    return {
        "PASSED": sum(item.outcome is SurvivalOutcome.PASSED for item in campaign.records.values()),
        "ELIMINATED": sum(item.outcome is SurvivalOutcome.ELIMINATED for item in campaign.records.values()),
        "BLOCKED_DEFERRED": sum(item.outcome in (SurvivalOutcome.BLOCKED, SurvivalOutcome.WAITING_FOR_RESOURCES)
                                for item in campaign.records.values()),
        "PENDING": sum(item.outcome in (SurvivalOutcome.PENDING, SurvivalOutcome.INTERRUPTED)
                       for item in campaign.records.values()),
    }


def survival_report_payload(campaign: SurvivalCampaign, manifest: TournamentManifest) -> dict[str, Any]:
    rows = []
    for candidate in manifest.candidates:
        record = campaign.records[candidate.candidate_id]
        provisioning = record.provisioning
        metrics = dict(record.metrics)
        rows.append({
            "candidate": candidate.candidate_id, "tag": candidate.variant.model,
            "provisioning_state": record.provisioning_state.value,
            "digest": None if provisioning is None else provisioning.digest,
            "package_size_bytes": None if provisioning is None else provisioning.size_bytes,
            "parameter_size": (candidate.variant.parameter_size if provisioning is None
                               else provisioning.parameter_size),
            "quantization": (candidate.variant.quantization if provisioning is None
                             else provisioning.quantization),
            "survival_outcome": record.outcome.value,
            "cold_load_latency_seconds": metrics.get("cold_load_latency_seconds"),
            "total_wall_time_seconds": metrics.get("wall_clock_duration_seconds"),
            "decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
            "ram_transient_delta_bytes": metrics.get("ram_transient_delta_bytes"),
            "vram_transient_delta_bytes": metrics.get("gpu_transient_vram_delta_bytes"),
            "max_gpu_temperature_c": metrics.get("max_gpu_temperature_c"),
            "cleanup_verified": record.cleanup_verified,
            "resource_recovery_verified": record.resource_recovery_verified,
            "default_server_identity_preserved": record.default_server_identity_preserved,
            "reason": record.reason,
        })
    return {
        "tournament": campaign_to_dict(campaign), "summary": _status_summary(campaign),
        "candidates": rows,
    }


def survival_report_text(campaign: SurvivalCampaign, manifest: TournamentManifest) -> str:
    summary = _status_summary(campaign)
    completed, total = campaign.progress
    lines = [
        "T/TEA Survival Tournament", "========================", f"Tournament: {campaign.tournament_id}",
        f"State: {campaign.state.value}", f"Progress: {completed}/{total}", "",
        "CANDIDATE                 PROVISIONING   SURVIVAL    COLD LOAD   DECODE TOK/S  CLEANUP",
    ]
    for row in survival_report_payload(campaign, manifest)["candidates"]:
        cold = "-" if row["cold_load_latency_seconds"] is None else f"{row['cold_load_latency_seconds']:.3f}s"
        rate = "-" if row["decode_tokens_per_second"] is None else f"{row['decode_tokens_per_second']:.2f}"
        cleanup = "-" if row["cleanup_verified"] is None else ("YES" if row["cleanup_verified"] else "NO")
        lines.append(f"{row['candidate']:<25} {row['provisioning_state']:<14} {row['survival_outcome']:<11} {cold:<11} {rate:<13} {cleanup}")
        if row["reason"]:
            lines.append(f"  {row['reason']}")
    lines.extend(("", f"PASSED: {summary['PASSED']}", f"ELIMINATED: {summary['ELIMINATED']}",
                  f"BLOCKED/DEFERRED: {summary['BLOCKED_DEFERRED']}", f"PENDING: {summary['PENDING']}"))
    return "\n".join(lines) + "\n"


def write_survival_report(store: SurvivalStore, campaign: SurvivalCampaign,
                          manifest: TournamentManifest) -> tuple[Any, Any]:
    json_path = store.tournaments.report_json_path(campaign.tournament_id)
    text_path = store.tournaments.report_text_path(campaign.tournament_id)
    TournamentStore._atomic_json(json_path, survival_report_payload(campaign, manifest))
    TournamentStore._atomic_text(text_path, survival_report_text(campaign, manifest))
    return json_path, text_path


class SurvivalTournamentEngine:
    """Run the closed survival manifest strictly one candidate at a time."""

    def __init__(self, tournament_id: str, store: SurvivalStore,
                 provider: OllamaTournamentProvider, *, now: Callable[[], datetime] = utc_now,
                 sleep: Callable[[float], None] = time.sleep,
                 max_runtime_attempts: int = 2) -> None:
        self.tournament_id, self.store, self.provider = tournament_id, store, provider
        self.now, self.sleep = now, sleep
        self.max_runtime_attempts = max_runtime_attempts
        self.active_candidates = 0

    def _save(self, campaign: SurvivalCampaign) -> SurvivalCampaign:
        latest = self.store.load(campaign.tournament_id)
        if latest.cancellation_requested and not campaign.cancellation_requested:
            campaign = replace(campaign, cancellation_requested=True)

        # Engine-owned timestamps must never regress behind persisted evidence.
        stamp = max(latest.updated_at, campaign.updated_at, self.now())
        heartbeat = campaign.heartbeat_at
        if heartbeat is not None and heartbeat < stamp:
            heartbeat = stamp
        campaign = replace(campaign, updated_at=stamp, heartbeat_at=heartbeat)
        return self.store.save(campaign)

    def _event(self, campaign: SurvivalCampaign, event: str, reason: str | None = None) -> None:
        self.store.append_event(campaign.tournament_id, event, self.now(), reason)

    def _record(self, campaign: SurvivalCampaign, record: CandidateSurvivalRecord, **changes: Any) -> SurvivalCampaign:
        records = dict(campaign.records)
        records[record.candidate_id] = record
        return self._save(replace(campaign, records=records, updated_at=self.now(), heartbeat_at=self.now(), **changes))

    def _cancelled(self, campaign: SurvivalCampaign) -> SurvivalCampaign | None:
        latest = self.store.load(campaign.tournament_id)
        if not latest.cancellation_requested:
            return None
        records = dict(latest.records)
        current = latest.current_candidate_id
        if current and records[current].outcome not in TERMINAL_CANDIDATE_OUTCOMES:
            records[current] = replace(records[current], outcome=SurvivalOutcome.CANCELLED,
                                       reason="tournament cancellation requested")
        cancelled = replace(latest, state=SurvivalCampaignState.CANCELLED, records=records,
                            updated_at=self.now(), heartbeat_at=self.now(), next_retry_at=None,
                            last_event="cancelled", last_reason="tournament cancellation requested")
        cancelled = self._save(cancelled)
        self._event(cancelled, "cancelled", cancelled.last_reason)
        write_survival_report(self.store, cancelled, self.provider.manifest)
        return cancelled

    def _wait(self, campaign: SurvivalCampaign, candidate_id: str, reason: str) -> SurvivalCampaign:
        policy = self.provider.execution_policy
        current = self.now()
        if current >= campaign.deadline_at:
            record = replace(campaign.records[candidate_id], outcome=SurvivalOutcome.BLOCKED,
                             reason="resource wait horizon elapsed")
            return self._record(campaign, record, state=SurvivalCampaignState.RUNNING,
                                next_retry_at=None, last_event="resource wait exhausted",
                                last_reason=record.reason)
        retry_at = current + timedelta(seconds=policy.resource_retry_seconds)
        record = replace(campaign.records[candidate_id], outcome=SurvivalOutcome.WAITING_FOR_RESOURCES,
                         reason=reason)
        waiting = self._record(campaign, record, state=SurvivalCampaignState.WAITING_FOR_RESOURCES,
                               next_retry_at=retry_at, last_event="waiting for resources", last_reason=reason)
        self._event(waiting, "waiting for resources", _event_reason(candidate_id, reason))
        while self.now() < retry_at:
            cancelled = self._cancelled(waiting)
            if cancelled is not None:
                return cancelled
            self.sleep(min(1.0, max(0.0, (retry_at - self.now()).total_seconds())))
        record = replace(waiting.records[candidate_id], outcome=SurvivalOutcome.PENDING)
        return self._record(waiting, record, state=SurvivalCampaignState.RUNNING,
                            next_retry_at=None, last_event="resource retry due", last_reason=reason)

    @staticmethod
    def _elimination_for(outcome: BenchmarkOutcome) -> EliminationReason:
        return {
            BenchmarkOutcome.TIMEOUT: EliminationReason.TIMEOUT,
            BenchmarkOutcome.THERMAL_LIMIT: EliminationReason.THERMAL_ABORT,
            BenchmarkOutcome.TELEMETRY_UNAVAILABLE: EliminationReason.POLICY_VIOLATION,
        }.get(outcome, EliminationReason.RUNTIME_FAILED)

    def run(self) -> SurvivalCampaign:
        campaign = self.store.load(self.tournament_id)
        manifest = self.provider.manifest
        tournament_round = manifest.round("survival_4k")
        task = manifest.task(tournament_round.task_ids[0])
        try:
            campaign = self._save(replace(campaign, state=SurvivalCampaignState.RUNNING,
                                          updated_at=self.now(), heartbeat_at=self.now(),
                                          current_round_id=tournament_round.round_id,
                                          last_event="survival round started", last_reason="sequential execution"))
            for candidate in manifest.candidates:
                while True:
                    cancelled = self._cancelled(campaign)
                    if cancelled is not None:
                        return cancelled
                    campaign = self.store.load(self.tournament_id)
                    record = campaign.records[candidate.candidate_id]
                    if record.outcome in TERMINAL_CANDIDATE_OUTCOMES:
                        break
                    self._event(campaign, "candidate selected", candidate.candidate_id)
                    record = replace(record, provisioning_state=ProvisioningState.INSPECTING,
                                     reason="inspecting exact manifest tag")
                    campaign = self._record(campaign, record, state=SurvivalCampaignState.RUNNING,
                                            current_candidate_id=candidate.candidate_id,
                                            last_event="candidate selected", last_reason=record.reason)
                    try:
                        evidence = self.provider.provision(
                            candidate, record.provisioning,
                            acquisition_preflight=self.provider.resource_manager.assess,
                        )
                    except ProvisioningResourceDelayed as exc:
                        campaign = self._wait(campaign, candidate.candidate_id, str(exc))
                        if campaign.state is SurvivalCampaignState.CANCELLED:
                            return campaign
                        continue
                    except ProvisioningDeferred as exc:
                        blocked = replace(record, provisioning_state=ProvisioningState.BLOCKED,
                                          outcome=SurvivalOutcome.BLOCKED, reason=str(exc))
                        campaign = self._record(campaign, blocked, state=SurvivalCampaignState.RUNNING,
                                                last_event="provisioning blocked", last_reason=str(exc))
                        self._event(campaign, "provisioning blocked", _event_reason(candidate.candidate_id, str(exc)))
                        break
                    except ProvisioningResourceDenied as exc:
                        eliminated = replace(record, provisioning_state=ProvisioningState.FAILED,
                                             outcome=SurvivalOutcome.ELIMINATED,
                                             reason=f"{EliminationReason.RESOURCE_DENIED.value}: {exc}")
                        campaign = self._record(campaign, eliminated, state=SurvivalCampaignState.RUNNING,
                                                last_event="candidate eliminated", last_reason=eliminated.reason)
                        self._event(campaign, "candidate eliminated", _event_reason(candidate.candidate_id, eliminated.reason))
                        break
                    except ProvisioningRejected as exc:
                        eliminated = replace(
                            record,
                            provisioning_state=ProvisioningState.FAILED,
                            outcome=SurvivalOutcome.ELIMINATED,
                            reason=f"{EliminationReason.POLICY_VIOLATION.value}: {exc}",
                        )
                        campaign = self._record(
                            campaign,
                            eliminated,
                            state=SurvivalCampaignState.RUNNING,
                            last_event="candidate eliminated",
                            last_reason=eliminated.reason,
                        )
                        self._event(
                            campaign,
                            "candidate eliminated",
                            _event_reason(candidate.candidate_id, eliminated.reason),
                        )
                        break
                    except Exception as exc:
                        eliminated = replace(
                            record,
                            provisioning_state=ProvisioningState.FAILED,
                            outcome=SurvivalOutcome.ELIMINATED,
                            reason=f"{EliminationReason.RUNTIME_FAILED.value}: {exc}",
                        )
                        campaign = self._record(
                            campaign,
                            eliminated,
                            state=SurvivalCampaignState.RUNNING,
                            last_event="candidate eliminated",
                            last_reason=eliminated.reason,
                        )
                        self._event(
                            campaign,
                            "candidate eliminated",
                            _event_reason(candidate.candidate_id, eliminated.reason),
                        )
                        break

                    record = replace(record, provisioning_state=ProvisioningState.VERIFIED,
                                     provisioning=evidence, outcome=SurvivalOutcome.PENDING,
                                     reason="exact model metadata verified")
                    campaign = self._record(campaign, record, state=SurvivalCampaignState.RUNNING,
                                            last_event="metadata verified", last_reason=record.reason)
                    self._event(campaign, "metadata verified", candidate.candidate_id)
                    resource_request = self.provider.resource_request(candidate, tournament_round, task)
                    assessment = self.provider.resource_manager.assess(resource_request)
                    if assessment.decision is ResourceDecision.DELAY:
                        campaign = self._wait(campaign, candidate.candidate_id, "; ".join(assessment.reasons))
                        if campaign.state is SurvivalCampaignState.CANCELLED:
                            return campaign
                        continue
                    if assessment.decision is ResourceDecision.DENY:
                        eliminated = replace(record, outcome=SurvivalOutcome.ELIMINATED,
                                             reason=f"RESOURCE_DENIED: {'; '.join(assessment.reasons)}")
                        campaign = self._record(campaign, eliminated, state=SurvivalCampaignState.RUNNING,
                                                last_event="candidate eliminated", last_reason=eliminated.reason)
                        self._event(campaign, "candidate eliminated", _event_reason(candidate.candidate_id, eliminated.reason))
                        break
                    selected_gpu_index = getattr(assessment, "selected_gpu_index", None)
                    resource_request = ResourceRequest(
                        workload_type=resource_request.workload_type,
                        estimated_ram_bytes=resource_request.estimated_ram_bytes,
                        estimated_disk_bytes=resource_request.estimated_disk_bytes,
                        estimated_gpu_vram_bytes=resource_request.estimated_gpu_vram_bytes,
                        gpu_index=selected_gpu_index,
                        id=resource_request.id,
                    )
                    execution_request = self.provider.build_execution_request(candidate, tournament_round, task, 0)
                    running = replace(record, outcome=SurvivalOutcome.RUNNING,
                                      attempt_count=record.attempt_count + 1, reason="survival execution started")
                    campaign = self._record(campaign, running, state=SurvivalCampaignState.RUNNING,
                                            last_event="survival started", last_reason=running.reason)
                    self._event(campaign, "survival started", candidate.candidate_id)
                    if self.active_candidates != 0:
                        raise RuntimeError("sequential invariant violated")
                    self.active_candidates = 1
                    try:
                        executed = self.provider.execute_survival(candidate, execution_request, resource_request)
                    finally:
                        self.active_candidates = 0
                    benchmark = executed.benchmark
                    common = {
                        "worker_pid": benchmark.worker_pid, "metrics": executed.metrics,
                        "cleanup_verified": executed.cleanup_verified,
                        "resource_recovery_verified": executed.resource_recovery_verified,
                        "default_server_identity_preserved": executed.default_server_identity_preserved,
                    }
                    if not executed.cleanup_verified:
                        reason = "CLEANUP_FAILED: required model/server/runtime cleanup was not verified"
                        completed = replace(running, outcome=SurvivalOutcome.ELIMINATED, reason=reason, **common)
                    elif not executed.default_server_identity_preserved:
                        reason = "POLICY_VIOLATION: default Ollama identity was not preserved"
                        completed = replace(running, outcome=SurvivalOutcome.ELIMINATED, reason=reason, **common)
                    elif not executed.resource_recovery_verified:
                        reason = "CLEANUP_FAILED: resource recovery remained outside safety tolerance"
                        completed = replace(running, outcome=SurvivalOutcome.ELIMINATED, reason=reason, **common)
                    elif benchmark.outcome is BenchmarkOutcome.COMPLETED and benchmark.worker_started:
                        reason = "bounded survival execution and all cleanup gates passed"
                        completed = replace(running, outcome=SurvivalOutcome.PASSED, reason=reason, **common)
                    elif benchmark.outcome is BenchmarkOutcome.RESOURCE_PRESSURE:
                        campaign = self._record(campaign, replace(running, outcome=SurvivalOutcome.PENDING,
                                                                  reason="; ".join(benchmark.reasons), **common),
                                                state=SurvivalCampaignState.RUNNING,
                                                last_event="runtime resource pressure", last_reason="; ".join(benchmark.reasons))
                        campaign = self._wait(campaign, candidate.candidate_id, "; ".join(benchmark.reasons))
                        continue
                    elif benchmark.outcome is BenchmarkOutcome.WORKER_FAILURE and running.attempt_count < self.max_runtime_attempts:
                        campaign = self._record(campaign, replace(running, outcome=SurvivalOutcome.INTERRUPTED,
                                                                  reason="; ".join(benchmark.reasons), **common),
                                                state=SurvivalCampaignState.RUNNING,
                                                last_event="runtime attempt failed", last_reason="; ".join(benchmark.reasons))
                        continue
                    else:
                        elimination = self._elimination_for(benchmark.outcome)
                        reason = f"{elimination.value}: {'; '.join(benchmark.reasons)}"
                        completed = replace(running, outcome=SurvivalOutcome.ELIMINATED, reason=reason, **common)
                    event = "candidate passed" if completed.outcome is SurvivalOutcome.PASSED else "candidate eliminated"
                    campaign = self._record(campaign, completed, state=SurvivalCampaignState.RUNNING,
                                            last_event=event, last_reason=completed.reason)
                    self._event(campaign, event, _event_reason(candidate.candidate_id, completed.reason))
                    self._event(campaign, "cleanup verified" if executed.cleanup_verified else "cleanup failed",
                                candidate.candidate_id)
                    write_survival_report(self.store, campaign, manifest)
                    break
            campaign = self.store.load(self.tournament_id)
            completed = replace(campaign, state=SurvivalCampaignState.COMPLETED,
                                current_candidate_id=None, updated_at=self.now(), heartbeat_at=self.now(),
                                next_retry_at=None, last_event="tournament completed",
                                last_reason="all manifest candidates reached a terminal survival disposition")
            completed = self._save(completed)
            self._event(completed, "tournament completed", completed.last_reason)
            write_survival_report(self.store, completed, manifest)
            return completed
        except KeyboardInterrupt:
            interrupted = replace(self.store.load(self.tournament_id), state=SurvivalCampaignState.INTERRUPTED,
                                  updated_at=self.now(), heartbeat_at=self.now(), next_retry_at=None,
                                  last_event="tournament interrupted", last_reason="supervisor interrupted")
            interrupted = self._save(interrupted)
            write_survival_report(self.store, interrupted, manifest)
            return interrupted
        except BaseException as exc:
            failed = replace(self.store.load(self.tournament_id), state=SurvivalCampaignState.FAILED,
                             updated_at=self.now(), heartbeat_at=self.now(), next_retry_at=None,
                             last_event="tournament failed", last_reason=f"{type(exc).__name__}: {exc}")
            self._save(failed)
            write_survival_report(self.store, failed, manifest)
            return failed

