"""Pure scoring and selection policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from .models import DIMENSIONS, SelectionRole, TournamentCandidate, TournamentResult, TournamentSelection


@dataclass(frozen=True)
class SelectionPolicy:
    minimum_capability: float = 0.4
    specialist_reasoning: float = 0.75
    specialist_margin: float = 0.1

    def __post_init__(self) -> None:
        for name in ("minimum_capability", "specialist_reasoning", "specialist_margin"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")


def composite(result: TournamentResult, weights: Mapping[str, float] | None = None) -> float | None:
    if not result.eligible or result.elimination is not None:
        return None
    weights = weights or {key: 1 for key in DIMENSIONS}
    if any(key not in DIMENSIONS or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for key, value in weights.items()):
        raise ValueError("invalid weights")
    required = [key for key, weight in weights.items() if weight > 0]
    if not required or any(key not in result.dimensions for key in required):
        return None
    total = sum(weights[key] for key in required)
    return sum(result.dimensions[key] * weights[key] for key in required) / total


def _fallback_ram_bytes(result: TournamentResult) -> float:
    value = result.metrics.get("peak_ram_bytes")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return float("inf")
    return float(value)


def select(
    results: Iterable[TournamentResult],
    candidates: Iterable[TournamentCandidate],
    policy: SelectionPolicy = SelectionPolicy(),
    *,
    eliminated_candidate_ids: Iterable[str] = (),
) -> tuple[TournamentSelection, ...]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    eliminated = frozenset(eliminated_candidate_ids)
    eligible = [
        result for result in results
        if result.candidate_id in by_id
        and result.candidate_id not in eliminated
        and by_id[result.candidate_id].enabled
        and not by_id[result.candidate_id].validation_only
        and result.eligible
        and result.elimination is None
        and (composite(result) or 0) >= policy.minimum_capability
    ]
    ordered = sorted(eligible, key=lambda item: (-(composite(item) or 0), item.candidate_id))
    primary = ordered[0] if ordered else None
    fallback = min(eligible, key=lambda item: (_fallback_ram_bytes(item), item.candidate_id), default=None)
    specialist = max(eligible, key=lambda item: (item.dimensions.get("reasoning", 0), item.candidate_id), default=None)
    if specialist and (
        specialist is primary
        or specialist.dimensions.get("reasoning", 0) < policy.specialist_reasoning
        or specialist.dimensions.get("reasoning", 0) < (primary.dimensions.get("reasoning", 0) + policy.specialist_margin if primary else 0)
    ):
        specialist = None
    return (
        TournamentSelection(SelectionRole.PRIMARY, primary.candidate_id if primary else None, "best eligible composite"),
        TournamentSelection(SelectionRole.LOW_RESOURCE_FALLBACK, fallback.candidate_id if fallback else None, "lowest-resource acceptable candidate"),
        TournamentSelection(SelectionRole.REASONING_SPECIALIST, specialist.candidate_id if specialist else None, "optional material reasoning advantage"),
    )
