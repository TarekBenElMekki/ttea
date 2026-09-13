"""Pure sequential planning, resource disposition, and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Iterable

from ttea.resources import ResourceDecision

from .manifest import ManifestError, TournamentManifest
from .models import (
    AttemptStatus, CandidateRoundResult, CandidateState, EliminationReason,
    EliminationRecord, MAX_CONCURRENCY, PersistedTournament, TaskResult,
    TournamentAttempt, TournamentPhase, TournamentState, TournamentWorkItem,
    require_transition,
)


class WorkDisposition(str, Enum):
    PROCEED = "PROCEED"
    DEFER = "DEFER"
    ELIMINATE = "ELIMINATE"


class TournamentIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class SchedulerDecision:
    disposition: WorkDisposition
    reason: str
    elimination_reason: EliminationReason | None = None


def schedule_resource_decision(decision: ResourceDecision, reason: str) -> SchedulerDecision:
    """Translate an existing admission decision without polling or execution."""
    if not isinstance(decision, ResourceDecision):
        raise ValueError("decision must be a ResourceDecision")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty")
    if decision is ResourceDecision.ALLOW:
        return SchedulerDecision(WorkDisposition.PROCEED, reason)
    if decision is ResourceDecision.DELAY:
        return SchedulerDecision(WorkDisposition.DEFER, reason)
    return SchedulerDecision(WorkDisposition.ELIMINATE, reason, EliminationReason.RESOURCE_DENIED)


def apply_scheduler_decision(
    tournament: PersistedTournament,
    work: TournamentWorkItem,
    decision: SchedulerDecision,
    decided_at: datetime,
) -> PersistedTournament:
    """Persist the conceptual scheduler effect without executing any work."""
    common = {
        "current_candidate_id": work.candidate_id,
        "current_round_id": work.round_id,
        "current_task_id": work.task_id,
        "current_repetition": work.repetition,
        "updated_at": decided_at,
    }
    if decision.disposition is WorkDisposition.PROCEED:
        if tournament.state is not TournamentState.RUNNING:
            require_transition(tournament.state, TournamentState.RUNNING)
        return replace(
            tournament, **common, state=TournamentState.RUNNING,
            resource_wait_started_at=None, resource_wait_reason=None,
            last_event="resource admission allowed", last_reason=decision.reason,
        )
    if decision.disposition is WorkDisposition.DEFER:
        if tournament.state is not TournamentState.WAITING_FOR_RESOURCES:
            require_transition(tournament.state, TournamentState.WAITING_FOR_RESOURCES)
        attempt_number = 1 + max((item.attempt_number for item in tournament.attempts if item.work_id == work.stable_id), default=0)
        deferred = replace(TournamentAttempt.new(work.stable_id, attempt_number), status=AttemptStatus.DEFERRED, reason=decision.reason)
        return replace(
            tournament, **common, state=TournamentState.WAITING_FOR_RESOURCES,
            attempts=tournament.attempts + (deferred,), resource_wait_started_at=decided_at,
            resource_wait_reason=decision.reason, last_event="work deferred for resources",
            last_reason=decision.reason,
        )
    record = EliminationRecord(
        candidate_id=work.candidate_id,
        reason=decision.elimination_reason or EliminationReason.RESOURCE_DENIED,
        timestamp=decided_at,
        explanation=decision.reason,
        round_id=work.round_id,
        task_id=work.task_id,
        work_id=work.stable_id,
    )
    return eliminate_candidate(replace(tournament, **common), record)


def ordered_work(manifest: TournamentManifest) -> tuple[TournamentWorkItem, ...]:
    """Candidate, round, task, then repetition order; concurrency is always one."""
    if MAX_CONCURRENCY != 1:  # pragma: no cover - protects the public invariant
        raise RuntimeError("tournament concurrency contract was changed")
    work: list[TournamentWorkItem] = []
    for candidate in manifest.candidates:
        if not candidate.enabled:
            continue
        for tournament_round in manifest.rounds:
            if candidate.validation_only and tournament_round.phase in {TournamentPhase.CAPABILITY, TournamentPhase.FINALIST}:
                continue
            for task_id in tournament_round.task_ids:
                task = manifest.task(task_id)
                for repetition in range(tournament_round.repetitions * task.repetitions):
                    work.append(TournamentWorkItem(candidate.candidate_id, tournament_round.round_id, task_id, repetition))
    return tuple(work)


def eligible_work(manifest: TournamentManifest, tournament: PersistedTournament) -> tuple[TournamentWorkItem, ...]:
    eliminated = {record.candidate_id for record in tournament.eliminations}
    eliminated.update(candidate_id for candidate_id, state in tournament.candidate_states.items() if state is CandidateState.ELIMINATED)
    eligible: list[TournamentWorkItem] = []
    for item in ordered_work(manifest):
        state = tournament.candidate_states.get(item.candidate_id, CandidateState.PENDING)
        tournament_round = manifest.round(item.round_id)
        if item.candidate_id in eliminated:
            continue
        if tournament_round.phase is TournamentPhase.FINALIST and state not in {CandidateState.FINALIST, CandidateState.SELECTED}:
            continue
        eligible.append(item)
    return tuple(eligible)


def next_work(manifest: TournamentManifest, tournament: PersistedTournament) -> TournamentWorkItem | None:
    """Return the first unfinished eligible item; deferred attempts do not block it."""
    for item in eligible_work(manifest, tournament):
        if item.stable_id not in tournament.results:
            return item
    return None


def calculate_progress(manifest: TournamentManifest, tournament: PersistedTournament) -> tuple[int, int]:
    known = {item.stable_id for item in ordered_work(manifest)}
    completed = known.intersection(tournament.results)
    remaining = {item.stable_id for item in eligible_work(manifest, tournament)} - completed
    return len(completed), len(completed) + len(remaining)


def with_calculated_progress(manifest: TournamentManifest, tournament: PersistedTournament) -> PersistedTournament:
    completed, total = calculate_progress(manifest, tournament)
    return replace(tournament, progress_completed=completed, progress_total=total)


def eliminate_candidate(tournament: PersistedTournament, record: EliminationRecord) -> PersistedTournament:
    states = dict(tournament.candidate_states)
    states[record.candidate_id] = CandidateState.ELIMINATED
    return replace(
        tournament,
        candidate_states=states,
        eliminations=tournament.eliminations + (record,),
        updated_at=record.timestamp,
        last_event="candidate eliminated",
        last_reason=record.explanation,
    )


def aggregate_round(
    candidate_id: str,
    round_id: str,
    result_ids: Iterable[str],
    results: dict[str, TaskResult] | object,
    *,
    eligible: bool,
    elimination: EliminationReason | None = None,
) -> CandidateRoundResult:
    ids = tuple(result_ids)
    mapping = dict(results)  # type: ignore[arg-type]
    try:
        selected = tuple(mapping[result_id] for result_id in ids)
    except KeyError as exc:
        raise ManifestError(f"aggregate references unknown task result: {exc.args[0]}") from exc
    if any(item.candidate_id != candidate_id or item.round_id != round_id for item in selected):
        raise ValueError("aggregate task result identity mismatch")
    dimensions: dict[str, float] = {}
    keys = {key for item in selected for key in item.dimensions}
    for key in keys:
        values = [item.dimensions[key] for item in selected if key in item.dimensions]
        dimensions[key] = sum(values) / len(values)
    metric_keys = {key for item in selected for key in item.metrics}
    metrics = {key: next((item.metrics[key] for item in reversed(selected) if key in item.metrics), None) for key in metric_keys}
    statuses = {item.status for item in selected}
    status = AttemptStatus.COMPLETED if statuses == {AttemptStatus.COMPLETED} else AttemptStatus.FAILED
    return CandidateRoundResult(candidate_id, round_id, ids, status, dimensions, metrics, eligible, elimination)


def validate_tournament_against_manifest(tournament: PersistedTournament, manifest: TournamentManifest) -> None:
    """Reject persisted identities that are not part of the closed manifest."""
    candidates = {item.candidate_id: item for item in manifest.candidates}
    rounds = {item.round_id: item for item in manifest.rounds}
    tasks = {item.task_id: item for item in manifest.tasks}
    work = {item.stable_id: item for item in ordered_work(manifest)}

    if set(tournament.candidate_states) != set(candidates):
        raise TournamentIntegrityError("candidate states must cover exactly the manifest candidates")

    def require_candidate(candidate_id: str, label: str) -> None:
        if candidate_id not in candidates:
            raise TournamentIntegrityError(f"{label} references unknown candidate: {candidate_id}")

    def require_round(round_id: str, label: str) -> None:
        if round_id not in rounds:
            raise TournamentIntegrityError(f"{label} references unknown round: {round_id}")

    def require_task(task_id: str, label: str) -> None:
        if task_id not in tasks:
            raise TournamentIntegrityError(f"{label} references unknown task: {task_id}")

    for candidate_id in tournament.candidate_states:
        require_candidate(candidate_id, "candidate state")
    if tournament.current_candidate_id is not None:
        require_candidate(tournament.current_candidate_id, "cursor")
    if tournament.current_round_id is not None:
        require_round(tournament.current_round_id, "cursor")
    if tournament.current_task_id is not None:
        require_task(tournament.current_task_id, "cursor")
    if tournament.current_round_id and tournament.current_task_id and tournament.current_task_id not in rounds[tournament.current_round_id].task_ids:
        raise TournamentIntegrityError("current task does not belong to current round")
    cursor = (tournament.current_candidate_id, tournament.current_round_id, tournament.current_task_id, tournament.current_repetition)
    if all(value is not None for value in cursor):
        cursor_id = TournamentWorkItem(cursor[0], cursor[1], cursor[2], cursor[3]).stable_id  # type: ignore[arg-type]
        if cursor_id not in work:
            raise TournamentIntegrityError("current repetition is not a valid manifest work item")
    for result_id, result in tournament.results.items():
        if result_id not in work or result.stable_id != result_id:
            raise TournamentIntegrityError(f"result is not a known manifest work item: {result_id}")
        require_candidate(result.candidate_id, "result")
        require_round(result.round_id, "result")
        require_task(result.task_id, "result")
        if result.task_id not in rounds[result.round_id].task_ids:
            raise TournamentIntegrityError("result task does not belong to result round")
    for attempt in tournament.attempts:
        if attempt.work_id not in work:
            raise TournamentIntegrityError(f"attempt references unknown work: {attempt.work_id}")
    for aggregate in tournament.round_results:
        require_candidate(aggregate.candidate_id, "round aggregate")
        require_round(aggregate.round_id, "round aggregate")
        for result_id in aggregate.task_result_ids:
            result = tournament.results.get(result_id)
            if result is None:
                raise TournamentIntegrityError(f"round aggregate references missing result: {result_id}")
            if result.candidate_id != aggregate.candidate_id or result.round_id != aggregate.round_id:
                raise TournamentIntegrityError("round aggregate result identity mismatch")
    for elimination in tournament.eliminations:
        require_candidate(elimination.candidate_id, "elimination")
        if tournament.candidate_states.get(elimination.candidate_id) is not CandidateState.ELIMINATED:
            raise TournamentIntegrityError("eliminated candidate must have ELIMINATED candidate state")
        if elimination.round_id is not None:
            require_round(elimination.round_id, "elimination")
        if elimination.task_id is not None:
            require_task(elimination.task_id, "elimination")
        if elimination.round_id and elimination.task_id and elimination.task_id not in rounds[elimination.round_id].task_ids:
            raise TournamentIntegrityError("elimination task does not belong to elimination round")
        if elimination.work_id is not None and elimination.work_id not in work:
            raise TournamentIntegrityError("elimination references unknown work")
        if elimination.work_id is not None:
            eliminated_work = work[elimination.work_id]
            if eliminated_work.candidate_id != elimination.candidate_id:
                raise TournamentIntegrityError("elimination work belongs to a different candidate")
            if elimination.round_id is not None and eliminated_work.round_id != elimination.round_id:
                raise TournamentIntegrityError("elimination work belongs to a different round")
            if elimination.task_id is not None and eliminated_work.task_id != elimination.task_id:
                raise TournamentIntegrityError("elimination work belongs to a different task")
    eliminated = {item.candidate_id for item in tournament.eliminations}
    eliminated.update(key for key, value in tournament.candidate_states.items() if value is CandidateState.ELIMINATED)
    for selection in tournament.selections:
        if selection.candidate_id is None:
            continue
        require_candidate(selection.candidate_id, "selection")
        candidate = candidates[selection.candidate_id]
        if not candidate.enabled or candidate.validation_only or selection.candidate_id in eliminated:
            raise TournamentIntegrityError("selection references an ineligible candidate")
