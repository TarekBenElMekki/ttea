"""Strict, versioned serialization for persisted tournament state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, TypeVar

from .models import (
    AttemptStatus, CandidateRoundResult, CandidateState, EliminationReason,
    EliminationRecord, PersistedTournament, SCHEMA_VERSION, SelectionRole,
    TaskResult, TournamentAttempt, TournamentPhase, TournamentSelection,
    TournamentState, thaw_json,
)

T = TypeVar("T")


class TournamentSerializationError(ValueError):
    pass


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_timestamp(value: object, name: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TournamentSerializationError(f"{name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TournamentSerializationError(f"malformed {name}") from exc
    return parsed


def _enum(enum_type: Callable[[object], T], value: object, name: str, *, optional: bool = False) -> T | None:
    if value is None and optional:
        return None
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise TournamentSerializationError(f"invalid {name}: {value!r}") from exc


def _attempt_to_dict(item: TournamentAttempt) -> dict[str, Any]:
    return {
        "attempt_id": item.attempt_id, "work_id": item.work_id,
        "attempt_number": item.attempt_number, "status": item.status.value,
        "started_at": _timestamp(item.started_at), "finished_at": _timestamp(item.finished_at),
        "reason": item.reason,
    }


def _result_to_dict(item: TaskResult) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id, "round_id": item.round_id,
        "task_id": item.task_id, "repetition": item.repetition,
        "status": item.status.value, "dimensions": dict(item.dimensions),
        "metrics": thaw_json(item.metrics),
        "elimination": item.elimination.value if item.elimination else None,
    }


def _round_result_to_dict(item: CandidateRoundResult) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id, "round_id": item.round_id,
        "task_result_ids": list(item.task_result_ids), "status": item.status.value,
        "dimensions": dict(item.dimensions), "metrics": thaw_json(item.metrics),
        "eligible": item.eligible,
        "elimination": item.elimination.value if item.elimination else None,
    }


def _elimination_to_dict(item: EliminationRecord) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id, "reason": item.reason.value,
        "timestamp": _timestamp(item.timestamp), "explanation": item.explanation,
        "round_id": item.round_id, "task_id": item.task_id, "work_id": item.work_id,
    }


def _selection_to_dict(item: TournamentSelection) -> dict[str, Any]:
    return {"role": item.role.value, "candidate_id": item.candidate_id, "reason": item.reason}


def tournament_to_dict(item: PersistedTournament) -> dict[str, Any]:
    """Convert validated state to deterministic JSON-compatible data."""
    return {
        "schema_version": item.schema_version, "tournament_id": item.tournament_id,
        "manifest_digest": item.manifest_digest, "manifest_snapshot": item.manifest_snapshot,
        "state": item.state.value, "created_at": _timestamp(item.created_at), "updated_at": _timestamp(item.updated_at),
        "current_phase": item.current_phase.value if item.current_phase else None,
        "current_round_id": item.current_round_id, "current_candidate_id": item.current_candidate_id,
        "current_task_id": item.current_task_id, "current_repetition": item.current_repetition,
        "progress_completed": item.progress_completed, "progress_total": item.progress_total,
        "resource_wait_started_at": _timestamp(item.resource_wait_started_at),
        "resource_wait_reason": item.resource_wait_reason,
        "candidate_states": {key: value.value for key, value in sorted(item.candidate_states.items())},
        "attempts": [_attempt_to_dict(value) for value in item.attempts],
        "results": {key: _result_to_dict(value) for key, value in sorted(item.results.items())},
        "round_results": [_round_result_to_dict(value) for value in item.round_results],
        "eliminations": [_elimination_to_dict(value) for value in item.eliminations],
        "selections": [_selection_to_dict(value) for value in item.selections],
        "last_event": item.last_event, "last_reason": item.last_reason,
    }


def tournament_from_dict(value: object) -> PersistedTournament:
    """Parse state strictly; malformed or incompatible data never becomes authoritative."""
    if not isinstance(value, dict):
        raise TournamentSerializationError("tournament state must be an object")
    try:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise TournamentSerializationError(f"unsupported schema_version: {value.get('schema_version')!r}")
        raw_states = value.get("candidate_states", {})
        raw_attempts = value.get("attempts", [])
        raw_results = value.get("results", {})
        raw_round_results = value.get("round_results", [])
        raw_eliminations = value.get("eliminations", [])
        raw_selections = value.get("selections", [])
        if not isinstance(raw_states, dict) or not isinstance(raw_attempts, list) or not isinstance(raw_results, dict):
            raise TournamentSerializationError("invalid tournament collections")
        attempts = tuple(TournamentAttempt(
            attempt_id=item["attempt_id"], work_id=item["work_id"], attempt_number=item["attempt_number"],
            status=_enum(AttemptStatus, item["status"], "attempt status"),
            started_at=_parse_timestamp(item.get("started_at"), "attempt started_at", optional=True),
            finished_at=_parse_timestamp(item.get("finished_at"), "attempt finished_at", optional=True), reason=item.get("reason"),
        ) for item in raw_attempts)
        results = {key: TaskResult(
            candidate_id=item["candidate_id"], round_id=item["round_id"], task_id=item["task_id"], repetition=item["repetition"],
            status=_enum(AttemptStatus, item["status"], "result status"), dimensions=item.get("dimensions", {}), metrics=item.get("metrics", {}),
            elimination=_enum(EliminationReason, item.get("elimination"), "result elimination", optional=True),
        ) for key, item in raw_results.items()}
        round_results = tuple(CandidateRoundResult(
            candidate_id=item["candidate_id"], round_id=item["round_id"], task_result_ids=tuple(item["task_result_ids"]),
            status=_enum(AttemptStatus, item["status"], "round status"), dimensions=item.get("dimensions", {}), metrics=item.get("metrics", {}),
            eligible=item["eligible"], elimination=_enum(EliminationReason, item.get("elimination"), "round elimination", optional=True),
        ) for item in raw_round_results)
        eliminations = tuple(EliminationRecord(
            candidate_id=item["candidate_id"], reason=_enum(EliminationReason, item["reason"], "elimination reason"),
            timestamp=_parse_timestamp(item["timestamp"], "elimination timestamp"), explanation=item["explanation"],
            round_id=item.get("round_id"), task_id=item.get("task_id"), work_id=item.get("work_id"),
        ) for item in raw_eliminations)
        selections = tuple(TournamentSelection(
            role=_enum(SelectionRole, item["role"], "selection role"), candidate_id=item.get("candidate_id"), reason=item["reason"],
        ) for item in raw_selections)
        return PersistedTournament(
            schema_version=value["schema_version"], tournament_id=value["tournament_id"],
            manifest_digest=value["manifest_digest"], manifest_snapshot=value["manifest_snapshot"],
            state=_enum(TournamentState, value["state"], "state"),
            created_at=_parse_timestamp(value["created_at"], "created_at"), updated_at=_parse_timestamp(value["updated_at"], "updated_at"),
            current_phase=_enum(TournamentPhase, value.get("current_phase"), "current_phase", optional=True),
            current_round_id=value.get("current_round_id"), current_candidate_id=value.get("current_candidate_id"),
            current_task_id=value.get("current_task_id"), current_repetition=value.get("current_repetition"),
            progress_completed=value.get("progress_completed", 0), progress_total=value.get("progress_total", 0),
            resource_wait_started_at=_parse_timestamp(value.get("resource_wait_started_at"), "resource_wait_started_at", optional=True),
            resource_wait_reason=value.get("resource_wait_reason"),
            candidate_states={key: _enum(CandidateState, state, "candidate state") for key, state in raw_states.items()},
            attempts=attempts, results=results, round_results=round_results,
            eliminations=eliminations, selections=selections,
            last_event=value.get("last_event"), last_reason=value.get("last_reason"),
        )
    except TournamentSerializationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TournamentSerializationError(f"invalid tournament state: {exc}") from exc
