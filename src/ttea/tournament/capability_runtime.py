"""Deterministic Capability Round 2 execution over the supervised Ollama provider."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import time
from typing import Callable

from ttea.benchmarking import BenchmarkOutcome, OllamaGenerationResult
from ttea.capability.catalogue import CAPABILITY_FIXTURES
from ttea.capability.evaluator import EvaluationStatus, evaluate
from ttea.capability.models import CapabilityFixture, thaw_json
from ttea.resources import ResourceDecision, ResourceRequest

from .capability import (
    ROUND2_CANDIDATES,
    CandidateCapabilitySummary,
    CapabilityStoreError,
    CapabilityTerminalState,
    CapabilityWorkId,
    CapabilityWorkStore,
    RawCapabilityResponse,
    capability_work_order,
    rank_candidates,
    summarize_candidate,
    terminal_state,
)
from .manifest import TournamentManifest
from .models import TournamentCandidate
from .ollama_provider import (
    OllamaTournamentProvider,
    ProvisioningRuntimeDelayed,
    SURVIVAL_GPU_HEADROOM_BYTES,
    SURVIVAL_RAM_START_HEADROOM_BYTES,
    SurvivalExecutionResult,
)
from .runner import TournamentExecutionRequest


CAPABILITY_ROUND_ID = "capability_4k"
CAPABILITY_CONTEXT_LIMIT = 4096
CAPABILITY_OUTPUT_LIMIT = 512
CAPABILITY_TIMEOUT_SECONDS = 180.0
CAPABILITY_TEMPERATURE = 0.0
CAPABILITY_CPU_THREADS = 4
CAPABILITY_KEEP_ALIVE = 0
CAPABILITY_MAX_TECHNICAL_ATTEMPTS = 2
CAPABILITY_RESOURCE_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60
CAPABILITY_RESOURCE_RETRY_SECONDS = 30.0

RETRYABLE_TECHNICAL_OUTCOMES = frozenset((
    BenchmarkOutcome.RESOURCE_PRESSURE,
    BenchmarkOutcome.TIMEOUT,
    BenchmarkOutcome.WORKER_FAILURE,
    BenchmarkOutcome.TELEMETRY_UNAVAILABLE,
))


class CapabilityIntegrityError(RuntimeError):
    """Capability evidence or fixture integrity no longer permits execution."""


@dataclass(frozen=True)
class CapabilityCampaignResult:
    tournament_id: str
    summaries: tuple[CandidateCapabilitySummary, ...]
    ranking: tuple[CandidateCapabilitySummary, ...]
    terminal: str
    integrity_reason: str | None = None


def _presented_state(fixture: CapabilityFixture) -> dict[str, object]:
    state = thaw_json(fixture.input.state)
    if fixture.input.state_delta is not None:
        state.update(thaw_json(fixture.input.state_delta))
    return state


def render_capability_prompt(fixture: CapabilityFixture) -> str:
    """Render deterministic model-visible input without oracle/scoring leakage."""
    if not isinstance(fixture, CapabilityFixture):
        raise ValueError("fixture must be a CapabilityFixture")
    payload = {
        "instruction": (
            "Return exactly one JSON object and nothing else. "
            "Choose ACT only when an available action should be taken; otherwise choose NO_ACTION. "
            "Use only listed action and subgoal IDs. Do not include rationale or markdown."
        ),
        "mission": fixture.input.mission,
        "state": _presented_state(fixture),
        "constraints": [item.to_dict() for item in fixture.input.constraints],
        "policy": {
            "forbid": [item.to_dict() for item in fixture.input.policy_forbid],
            "require": [item.to_dict() for item in fixture.input.policy_require],
        },
        "actions": [item.to_dict() for item in fixture.input.actions],
        "subgoal_ids": [item.id for item in fixture.subgoals],
        "prior_result": (
            None if fixture.input.prior_result is None
            else fixture.input.prior_result.to_dict()
        ),
        "state_delta": (
            None if fixture.input.state_delta is None
            else thaw_json(fixture.input.state_delta)
        ),
        "output_schema": {
            "mode": "ACT|NO_ACTION",
            "plan": [{
                "subgoal": "listed_subgoal_id",
                "action": "listed_action_id",
                "depends_on": ["listed_subgoal_id"],
            }],
        },
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def capability_execution_request(
    candidate: TournamentCandidate,
    fixture: CapabilityFixture,
) -> TournamentExecutionRequest:
    return TournamentExecutionRequest(
        candidate.candidate_id,
        CAPABILITY_ROUND_ID,
        fixture.id,
        0,
        "ollama",
        candidate.variant.model,
        render_capability_prompt(fixture),
        CAPABILITY_CONTEXT_LIMIT,
        CAPABILITY_OUTPUT_LIMIT,
        CAPABILITY_TIMEOUT_SECONDS,
        CAPABILITY_TEMPERATURE,
        CAPABILITY_CPU_THREADS,
        CAPABILITY_KEEP_ALIVE,
    )


def capability_resource_request(candidate: TournamentCandidate) -> ResourceRequest:
    return ResourceRequest(
        workload_type=f"tournament-capability:{candidate.candidate_id}",
        estimated_ram_bytes=SURVIVAL_RAM_START_HEADROOM_BYTES,
        estimated_disk_bytes=0,
        estimated_gpu_vram_bytes=SURVIVAL_GPU_HEADROOM_BYTES,
    )


def _technical_evidence(
    assessment,
    execution: SurvivalExecutionResult | None,
    *,
    error: BaseException | None = None,
) -> dict[str, object]:
    benchmark = None if execution is None else execution.benchmark
    return {
        "admission": {
            "decision": assessment.decision.value,
            "reasons": list(assessment.reasons),
            "selected_gpu_index": getattr(assessment, "selected_gpu_index", None),
        },
        "benchmark": None if benchmark is None else {
            "outcome": benchmark.outcome.value,
            "reasons": list(benchmark.reasons),
            "started_at": benchmark.started_at.isoformat(),
            "finished_at": benchmark.finished_at.isoformat(),
            "worker_pid": benchmark.worker_pid,
            "worker_started": benchmark.worker_started,
            "worker_started_epoch": getattr(benchmark, "worker_started_epoch", None),
            "timeout_kind": benchmark.timeout_kind,
        },
        "metrics": {} if execution is None else dict(execution.metrics),
        "cleanup_verified": None if execution is None else execution.cleanup_verified,
        "resource_recovery_verified": None if execution is None else execution.resource_recovery_verified,
        "default_server_identity_preserved": (
            None if execution is None else execution.default_server_identity_preserved
        ),
        "owned_ollama_pid": None if execution is None else execution.owned_ollama_pid,
        "owned_ollama_started_epoch": (
            None if execution is None else execution.owned_ollama_started_epoch
        ),
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }


class CapabilityRuntimeEngine:
    """Execute the frozen 4x9 Capability matrix sequentially and durably."""

    def __init__(
        self,
        tournament_id: str,
        store: CapabilityWorkStore,
        manifest: TournamentManifest,
        provider: OllamaTournamentProvider,
        resource_manager,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(tournament_id, str) or not tournament_id.strip():
            raise ValueError("tournament_id must be non-empty")
        if tuple(candidate.candidate_id for candidate in manifest.candidates) != ROUND2_CANDIDATES:
            raise CapabilityIntegrityError("Capability manifest does not match frozen survivor order")
        self.tournament_id = tournament_id
        self.store = store
        self.manifest = manifest
        self.provider = provider
        self.resource_manager = resource_manager
        self.monotonic = monotonic
        self.sleep = sleep
        self._fixtures = {fixture.id: fixture for fixture in CAPABILITY_FIXTURES}

    def _candidate(self, candidate_id: str) -> TournamentCandidate:
        try:
            return self.manifest.candidate(candidate_id)
        except Exception as exc:
            raise CapabilityIntegrityError(
                f"candidate provenance lookup failed: {candidate_id}"
            ) from exc

    def _fixture(self, work: CapabilityWorkId) -> CapabilityFixture:
        fixture = self._fixtures.get(work.fixture_id)
        if fixture is None or fixture.version != work.fixture_version:
            raise CapabilityIntegrityError("Capability work references unknown fixture identity")
        return fixture

    def _evaluate_persisted(self, work: CapabilityWorkId, fixture: CapabilityFixture):
        raw = self.store.raw_response(work)
        result = evaluate(fixture, raw.response_utf8)
        if result.status is EvaluationStatus.FIXTURE_INVALID:
            self.store.mark_terminal(work, "BLOCKED_INTEGRITY", "fixture failed deterministic evaluation validation")
            raise CapabilityIntegrityError("fixture failed deterministic evaluation validation")
        self.store.publish_result(work, result)
        self.store.mark_terminal(work, "EVALUATED")
        return self.store.load(work)

    def _recover_work(self, work: CapabilityWorkId, fixture: CapabilityFixture):
        record = self.store.create(work, fixture.digest)
        if record.terminal_status:
            return record
        if record.active_attempt_id:
            record = self.store.recover(work)
        if record.evaluator_result_digest:
            return self.store.mark_terminal(work, "EVALUATED")
        if record.raw_response_digest:
            if not record.technical_evidence_refs:
                return self.store.mark_terminal(
                    work,
                    "TECHNICAL_INCOMPLETE",
                    "raw response exists without finalized cleanup evidence",
                )
            return self._evaluate_persisted(work, fixture)
        if record.technical_attempt_count >= CAPABILITY_MAX_TECHNICAL_ATTEMPTS:
            return self.store.mark_terminal(
                work,
                "TECHNICAL_INCOMPLETE",
                "bounded technical attempts exhausted before a durable response",
            )
        return record

    def _assessment(self, request: ResourceRequest):
        deadline = self.monotonic() + CAPABILITY_RESOURCE_WAIT_TIMEOUT_SECONDS
        while True:
            assessment = self.resource_manager.assess(request)
            if assessment.decision is not ResourceDecision.DELAY:
                return assessment
            if self.monotonic() >= deadline:
                return assessment
            self.sleep(CAPABILITY_RESOURCE_RETRY_SECONDS)

    def run_work(self, work: CapabilityWorkId):
        fixture = self._fixture(work)
        record = self._recover_work(work, fixture)
        if record.terminal_status:
            return record

        candidate = self._candidate(work.candidate_id)
        while True:
            request = capability_resource_request(candidate)
            assessment = self._assessment(request)
            if assessment.decision is ResourceDecision.DELAY:
                return self.store.mark_terminal(
                    work,
                    "TECHNICAL_INCOMPLETE",
                    "resource wait deadline exhausted before admission",
                )
            if assessment.decision is ResourceDecision.DENY:
                return self.store.mark_terminal(
                    work,
                    "TECHNICAL_INCOMPLETE",
                    "; ".join(assessment.reasons),
                )

            selected_request = replace(
                request,
                gpu_index=getattr(assessment, "selected_gpu_index", None),
            )
            active = self.store.begin_attempt(work)
            attempt_id = active.active_attempt_id
            execution = None
            try:
                execution = self.provider.execute_prompt(
                    candidate,
                    capability_execution_request(candidate, fixture),
                    selected_request,
                )
            except ProvisioningRuntimeDelayed as exc:
                evidence_ref = self.store.publish_technical_evidence(
                    work,
                    attempt_id,
                    _technical_evidence(assessment, None, error=exc),
                )
                record = self.store.finish_attempt(work, evidence_ref)
                if record.technical_attempt_count < CAPABILITY_MAX_TECHNICAL_ATTEMPTS:
                    continue
                return self.store.mark_terminal(
                    work,
                    "TECHNICAL_INCOMPLETE",
                    "owned runtime startup remained unavailable",
                )
            except Exception as exc:
                evidence_ref = self.store.publish_technical_evidence(
                    work,
                    attempt_id,
                    _technical_evidence(assessment, None, error=exc),
                )
                self.store.finish_attempt(work, evidence_ref)
                return self.store.mark_terminal(
                    work,
                    "TECHNICAL_INCOMPLETE",
                    f"provider execution failed: {type(exc).__name__}: {exc}",
                )

            generated = execution.benchmark.worker_result
            raw = None
            if isinstance(generated, OllamaGenerationResult) and isinstance(generated.response, str):
                raw = RawCapabilityResponse.create(
                    work,
                    fixture.digest,
                    attempt_id,
                    generated.response,
                )
                self.store.publish_raw(raw)

            evidence_ref = self.store.publish_technical_evidence(
                work,
                attempt_id,
                _technical_evidence(assessment, execution),
            )
            record = self.store.finish_attempt(work, evidence_ref)

            cleanup_ok = (
                execution.cleanup_verified
                and execution.resource_recovery_verified
                and execution.default_server_identity_preserved
            )

            if raw is not None:
                if execution.benchmark.outcome is BenchmarkOutcome.COMPLETED and cleanup_ok:
                    return self._evaluate_persisted(work, fixture)
                return self.store.mark_terminal(
                    work,
                    "TECHNICAL_INCOMPLETE",
                    "generation produced a durable response but supervised cleanup or execution failed",
                )

            if (
                execution.benchmark.outcome in RETRYABLE_TECHNICAL_OUTCOMES
                and cleanup_ok
                and record.technical_attempt_count < CAPABILITY_MAX_TECHNICAL_ATTEMPTS
            ):
                continue

            return self.store.mark_terminal(
                work,
                "TECHNICAL_INCOMPLETE",
                "technical execution ended without a durable capability response",
            )

    def _summary(self) -> tuple[CandidateCapabilitySummary, ...]:
        summaries = []
        for candidate_id in ROUND2_CANDIDATES:
            results = {}
            incomplete = False
            for fixture in CAPABILITY_FIXTURES:
                work = CapabilityWorkId(
                    self.tournament_id,
                    candidate_id,
                    fixture.id,
                    fixture.version,
                )
                record = self.store.load(work)
                if record.evaluator_result_digest:
                    results[fixture.id] = self.store.evaluator_result(work)
                else:
                    incomplete = True
            summaries.append(
                summarize_candidate(candidate_id, results, incomplete=incomplete)
            )
        return tuple(summaries)

    def run(self) -> CapabilityCampaignResult:
        integrity_reason = None
        try:
            for work in capability_work_order(self.tournament_id):
                self.run_work(work)
        except (CapabilityIntegrityError, CapabilityStoreError) as exc:
            integrity_reason = str(exc)

        summaries = self._summary()
        ranked = rank_candidates(summaries)
        terminal = terminal_state(
            summaries,
            integrity_blocker=integrity_reason is not None,
        )
        return CapabilityCampaignResult(
            self.tournament_id,
            summaries,
            ranked,
            terminal,
            integrity_reason,
        )