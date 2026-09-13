"""Structured, non-fabricating tournament reports."""

from __future__ import annotations

from typing import Any

from .manifest import TournamentManifest
from .models import DIMENSIONS, PERFORMANCE_METRICS, PersistedTournament, TournamentResult, thaw_json
from .scoring import composite
from .store import TournamentStore

METRIC_FIELDS = PERFORMANCE_METRICS


def report_payload(tournament: PersistedTournament, manifest: TournamentManifest | None = None) -> dict[str, Any]:
    candidate_ids = [candidate.candidate_id for candidate in manifest.candidates] if manifest else list(tournament.candidate_states)
    aggregates = []
    scores_by_candidate: dict[str, list[float]] = {}
    for aggregate in tournament.round_results:
        dimensions = {key: aggregate.dimensions.get(key) for key in sorted(DIMENSIONS)}
        metrics = {key: thaw_json(aggregate.metrics.get(key)) for key in METRIC_FIELDS}
        metrics.update({key: thaw_json(value) for key, value in aggregate.metrics.items() if key not in metrics})
        score = composite(TournamentResult(aggregate.candidate_id, aggregate.dimensions, aggregate.metrics, aggregate.eligible, aggregate.elimination))
        coverage = sum(key in aggregate.dimensions for key in DIMENSIONS) / len(DIMENSIONS)
        aggregates.append({
            "candidate_id": aggregate.candidate_id, "round_id": aggregate.round_id,
            "status": aggregate.status.value, "eligible": aggregate.eligible,
            "elimination": aggregate.elimination.value if aggregate.elimination else None,
            "task_result_ids": list(aggregate.task_result_ids),
            "dimensions": dimensions, "metrics": metrics,
            "composite": score, "score_coverage": coverage,
        })
        if score is not None:
            scores_by_candidate.setdefault(aggregate.candidate_id, []).append(score)
    ranking = sorted(
        ({"candidate_id": candidate_id, "score": sum(values) / len(values)} for candidate_id, values in scores_by_candidate.items()),
        key=lambda item: (-item["score"], item["candidate_id"]),
    )
    return {
        "tournament_id": tournament.tournament_id,
        "state": tournament.state.value,
        "manifest_digest": tournament.manifest_digest,
        "created_at": tournament.created_at.isoformat(),
        "updated_at": tournament.updated_at.isoformat(),
        "progress": {"completed": tournament.progress_completed, "total": tournament.progress_total},
        "candidate_states": {candidate_id: tournament.candidate_states.get(candidate_id).value if tournament.candidate_states.get(candidate_id) else None for candidate_id in candidate_ids},
        "ranking": ranking,
        "round_aggregates": aggregates,
        "task_result_references": sorted(tournament.results),
        "eliminations": [{
            "candidate_id": item.candidate_id, "reason": item.reason.value,
            "timestamp": item.timestamp.isoformat(), "explanation": item.explanation,
            "round_id": item.round_id, "task_id": item.task_id, "work_id": item.work_id,
        } for item in tournament.eliminations],
        "selections": [{"role": item.role.value, "candidate_id": item.candidate_id, "reason": item.reason} for item in tournament.selections],
    }


def text_report(tournament: PersistedTournament, manifest: TournamentManifest | None = None) -> str:
    payload = report_payload(tournament, manifest)
    lines = [
        "T/TEA Tournament", "================", "", f"Tournament: {tournament.tournament_id}",
        f"State: {tournament.state.value}", f"Progress: {tournament.progress_completed}/{tournament.progress_total}", "", "Candidates:",
    ]
    states = payload["candidate_states"]
    lines.extend(f"- {candidate_id}: {state or 'unknown'}" for candidate_id, state in states.items())
    if not states:
        lines.append("- (none)")
    lines.extend(["", "Eliminated:"])
    lines.extend(f"- {item['candidate_id']}: {item['reason']} - {item['explanation']}" for item in payload["eliminations"])
    if not payload["eliminations"]:
        lines.append("- (none)")
    lines.extend(["", "Scores:"])
    lines.extend(f"- {item['candidate_id']}: {item['score']:.6f}" for item in payload["ranking"])
    if not payload["ranking"]:
        lines.append("- (none)")
    by_role = {item["role"]: item for item in payload["selections"]}
    lines.extend(["", "Selections:"])
    for label, role in (("Primary", "PRIMARY"), ("Fallback", "LOW_RESOURCE_FALLBACK"), ("Reasoning specialist", "REASONING_SPECIALIST")):
        selected = by_role.get(role)
        lines.append(f"{label}: {selected['candidate_id'] if selected and selected['candidate_id'] else '(none)'}")
    return "\n".join(lines) + "\n"


def write_report(store: TournamentStore, tournament: PersistedTournament, manifest: TournamentManifest | None = None) -> tuple[object, object]:
    payload = report_payload(tournament, manifest)
    TournamentStore._atomic_json(store.report_json_path(tournament.tournament_id), payload)
    TournamentStore._atomic_text(store.report_text_path(tournament.tournament_id), text_report(tournament, manifest))
    return store.report_json_path(tournament.tournament_id), store.report_text_path(tournament.tournament_id)
