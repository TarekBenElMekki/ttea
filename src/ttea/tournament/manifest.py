"""Closed-world deterministic tournament manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .models import CandidateVariant, TournamentCandidate, TournamentPhase, TournamentRound, TournamentTask


class ManifestError(ValueError):
    pass


class ManifestMismatchError(ManifestError):
    pass


SAFETY_MODEL_ID = "smollm2-safety"
SAFETY_CANDIDATE = TournamentCandidate(
    candidate_id=SAFETY_MODEL_ID,
    family="smollm2",
    variant=CandidateVariant("ollama", "smollm2:135m-instruct-q4_0", "Q4_0", "135M"),
    estimated_disk_bytes=None,
    estimated_ram_bytes=0,
    estimated_vram_bytes=0,
    context_target=512,
    purpose="safety validation",
    validation_only=True,
)


class TournamentManifest:
    def __init__(self, candidates: Iterable[TournamentCandidate] = (), rounds: Iterable[TournamentRound] = (), tasks: Iterable[TournamentTask] = ()) -> None:
        self.candidates = tuple(candidates)
        self.rounds = tuple(rounds)
        self.tasks = tuple(tasks)
        self._unique(self.candidates, "candidate_id")
        self._unique(self.rounds, "round_id")
        self._unique(self.tasks, "task_id")
        identities = [candidate.canonical_identity for candidate in self.candidates]
        if len(identities) != len(set(identities)):
            raise ManifestError("duplicate canonical candidate identity")
        task_ids = {task.task_id for task in self.tasks}
        for tournament_round in self.rounds:
            if not set(tournament_round.task_ids) <= task_ids:
                raise ManifestError("round references unknown task")
            for task_id in tournament_round.task_ids:
                task = self.task(task_id)
                if tournament_round.context_limit > task.max_context or tournament_round.output_limit > task.max_output:
                    raise ManifestError("round limit exceeds task constraint")

    @staticmethod
    def _unique(items: tuple[Any, ...], attribute: str) -> None:
        values = [getattr(item, attribute) for item in items]
        if len(values) != len(set(values)):
            raise ManifestError(f"duplicate {attribute}")

    @staticmethod
    def _find(items: tuple[Any, ...], attribute: str, value: str) -> Any:
        for item in items:
            if getattr(item, attribute) == value:
                return item
        raise ManifestError(f"unknown {attribute}: {value}")

    def candidate(self, candidate_id: str) -> TournamentCandidate:
        return self._find(self.candidates, "candidate_id", candidate_id)

    def round(self, round_id: str) -> TournamentRound:
        return self._find(self.rounds, "round_id", round_id)

    def task(self, task_id: str) -> TournamentTask:
        return self._find(self.tasks, "task_id", task_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "family": item.family,
                    "provider": item.variant.provider,
                    "model": item.variant.model,
                    "quantization": item.variant.quantization,
                    "parameter_size": item.variant.parameter_size,
                    "estimated_disk_bytes": item.estimated_disk_bytes,
                    "estimated_ram_bytes": item.estimated_ram_bytes,
                    "estimated_vram_bytes": item.estimated_vram_bytes,
                    "context_target": item.context_target,
                    "purpose": item.purpose,
                    "enabled": item.enabled,
                    "validation_only": item.validation_only,
                }
                for item in self.candidates
            ],
            "rounds": [
                {
                    "round_id": item.round_id,
                    "phase": item.phase.value,
                    "task_ids": list(item.task_ids),
                    "repetitions": item.repetitions,
                    "context_limit": item.context_limit,
                    "output_limit": item.output_limit,
                    "resource_class": item.resource_class,
                    "timeout_policy": item.timeout_policy,
                    "scoring_policy": item.scoring_policy,
                    "elimination_policy": item.elimination_policy,
                }
                for item in self.rounds
            ],
            "tasks": [
                {
                    "task_id": item.task_id,
                    "category": item.category,
                    "prompt": item.prompt,
                    "output_contract": item.output_contract,
                    "scorer_type": item.scorer_type,
                    "weight": item.weight,
                    "max_context": item.max_context,
                    "max_output": item.max_output,
                    "timeout_seconds": item.timeout_seconds,
                    "repetitions": item.repetitions,
                }
                for item in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "TournamentManifest":
        if not isinstance(value, dict):
            raise ManifestError("manifest must be an object")
        try:
            candidates = tuple(
                TournamentCandidate(
                    candidate_id=item["candidate_id"], family=item["family"],
                    variant=CandidateVariant(item["provider"], item["model"], item.get("quantization"), item.get("parameter_size")),
                    estimated_disk_bytes=item.get("estimated_disk_bytes"), estimated_ram_bytes=item["estimated_ram_bytes"],
                    estimated_vram_bytes=item["estimated_vram_bytes"], context_target=item["context_target"], purpose=item["purpose"],
                    enabled=item.get("enabled", True), validation_only=item.get("validation_only", False),
                ) for item in value["candidates"]
            )
            rounds = tuple(TournamentRound(
                round_id=item["round_id"], phase=TournamentPhase(item["phase"]), task_ids=tuple(item["task_ids"]),
                repetitions=item["repetitions"], context_limit=item["context_limit"], output_limit=item["output_limit"],
                resource_class=item["resource_class"], timeout_policy=item["timeout_policy"], scoring_policy=item["scoring_policy"], elimination_policy=item["elimination_policy"],
            ) for item in value["rounds"])
            tasks = tuple(TournamentTask(**item) for item in value["tasks"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"invalid manifest: {exc}") from exc
        return cls(candidates, rounds, tasks)

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def verify_digest(self, digest: str) -> None:
        if self.digest != digest:
            raise ManifestMismatchError("manifest digest does not match persisted tournament")
