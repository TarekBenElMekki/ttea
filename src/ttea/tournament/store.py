"""Filesystem persistence for inert tournaments."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .manifest import ManifestMismatchError, TournamentManifest
from .models import (
    AttemptStatus, EliminationRecord, PersistedTournament, TaskResult,
    TournamentAttempt, TournamentSelection, TournamentState,
    require_attempt_transition, require_candidate_transition, require_transition,
    resume_interrupted as resume_interrupted_state,
)
from .planning import (
    calculate_progress, eliminate_candidate, validate_tournament_against_manifest,
    with_calculated_progress,
)
from .serialization import TournamentSerializationError, tournament_from_dict, tournament_to_dict


class TournamentStoreError(RuntimeError):
    pass


class ResultConflictError(TournamentStoreError):
    pass


class AttemptConflictError(TournamentStoreError):
    pass


class TournamentTransitionError(TournamentStoreError):
    pass


class AttemptTransitionError(TournamentStoreError):
    pass


class TimestampRegressionError(TournamentStoreError):
    pass


class CandidateTransitionError(TournamentStoreError):
    pass


class EvidenceConflictError(TournamentStoreError):
    pass


class TournamentStore:
    def __init__(self, root: Path | str = Path("data/tournaments")) -> None:
        self.root = Path(root)

    def tournament_path(self, tournament_id: str) -> Path:
        if not isinstance(tournament_id, str) or not tournament_id.strip() or Path(tournament_id).name != tournament_id:
            raise ValueError("invalid tournament_id")
        return self.root / tournament_id

    def state_path(self, tournament_id: str) -> Path:
        return self.tournament_path(tournament_id) / "tournament.json"

    def manifest_snapshot_path(self, tournament_id: str) -> Path:
        return self.tournament_path(tournament_id) / "manifest.json"

    def events_path(self, tournament_id: str) -> Path:
        return self.tournament_path(tournament_id) / "events.jsonl"

    def report_json_path(self, tournament_id: str) -> Path:
        return self.tournament_path(tournament_id) / "report.json"

    def report_text_path(self, tournament_id: str) -> Path:
        return self.tournament_path(tournament_id) / "report.txt"

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _atomic_json(cls, path: Path, value: Any) -> None:
        cls._atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

    @staticmethod
    def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise TournamentStoreError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    @classmethod
    def _read_json(cls, path: Path, description: str) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=cls._json_object)
        except TournamentStoreError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise TournamentStoreError(f"cannot read {description}: {exc}") from exc

    def create(self, tournament: PersistedTournament, manifest: TournamentManifest) -> PersistedTournament:
        if self.exists(tournament.tournament_id):
            raise TournamentStoreError(f"tournament already exists: {tournament.tournament_id}")
        if tournament.manifest_digest != manifest.digest:
            raise ManifestMismatchError("initial manifest does not match tournament digest")
        if Path(tournament.manifest_snapshot).name != "manifest.json":
            raise TournamentStoreError("manifest_snapshot must reference manifest.json")
        validate_tournament_against_manifest(tournament, manifest)
        if (tournament.progress_completed, tournament.progress_total) != calculate_progress(manifest, tournament):
            raise TournamentStoreError("persisted progress does not match manifest work")
        directory = self.tournament_path(tournament.tournament_id)
        directory.mkdir(parents=True, exist_ok=False)
        self._atomic_json(self.manifest_snapshot_path(tournament.tournament_id), manifest.to_dict())
        self._atomic_json(self.state_path(tournament.tournament_id), tournament_to_dict(tournament))
        return tournament

    def exists(self, tournament_id: str) -> bool:
        return self.state_path(tournament_id).is_file()

    def list(self) -> tuple[PersistedTournament, ...]:
        if not self.root.is_dir():
            return ()
        values = [self.load(path.name) for path in self.root.iterdir() if path.is_dir() and (path / "tournament.json").is_file()]
        return tuple(sorted(values, key=lambda item: (item.updated_at, item.tournament_id), reverse=True))

    def latest(self) -> PersistedTournament | None:
        values = self.list()
        return values[0] if values else None

    def load_manifest(self, tournament_id: str) -> TournamentManifest:
        path = self.manifest_snapshot_path(tournament_id)
        raw = self._read_json(path, "manifest snapshot")
        return TournamentManifest.from_dict(raw)

    def load(self, tournament_id: str) -> PersistedTournament:
        path = self.state_path(tournament_id)
        raw = self._read_json(path, "tournament state")
        try:
            tournament = tournament_from_dict(raw)
        except TournamentSerializationError as exc:
            raise TournamentStoreError(str(exc)) from exc
        if tournament.tournament_id != tournament_id:
            raise TournamentStoreError("stored tournament ID does not match directory")
        persisted_manifest = self.load_manifest(tournament_id)
        persisted_manifest.verify_digest(tournament.manifest_digest)
        validate_tournament_against_manifest(tournament, persisted_manifest)
        if (tournament.progress_completed, tournament.progress_total) != calculate_progress(persisted_manifest, tournament):
            raise TournamentStoreError("persisted progress does not match manifest work")
        return tournament

    def verify_manifest(self, tournament_id: str, manifest: TournamentManifest) -> None:
        manifest.verify_digest(self.load(tournament_id).manifest_digest)

    @staticmethod
    def _validate_replacement(previous: PersistedTournament, tournament: PersistedTournament, *, allow_resume: bool = False) -> None:
        if tournament.tournament_id != previous.tournament_id:
            raise TournamentStoreError("tournament_id cannot change")
        if tournament.schema_version != previous.schema_version:
            raise TournamentStoreError("schema_version cannot change")
        if tournament.manifest_digest != previous.manifest_digest:
            raise ManifestMismatchError("replacement state changed manifest digest")
        if tournament.updated_at < previous.updated_at:
            raise TimestampRegressionError("updated_at cannot move backwards")
        if tournament.state is not previous.state:
            if allow_resume and previous.state is TournamentState.INTERRUPTED and tournament.state is TournamentState.PAUSED:
                pass
            else:
                try:
                    require_transition(previous.state, tournament.state)
                except ValueError as exc:
                    raise TournamentTransitionError(f"illegal persisted tournament transition: {previous.state.value} -> {tournament.state.value}") from exc
        old_candidate_states = dict(previous.candidate_states)
        new_candidate_states = dict(tournament.candidate_states)
        if not old_candidate_states.keys() <= new_candidate_states.keys():
            raise CandidateTransitionError("persisted candidate states cannot be removed")
        if new_candidate_states.keys() != old_candidate_states.keys():
            raise CandidateTransitionError("candidate states cannot be inserted after creation")
        for candidate_id, old_state in old_candidate_states.items():
            new_state = new_candidate_states[candidate_id]
            if new_state is old_state:
                continue
            try:
                require_candidate_transition(old_state, new_state)
            except ValueError as exc:
                raise CandidateTransitionError(
                    f"illegal persisted candidate transition: {candidate_id}: {old_state.value} -> {new_state.value}"
                ) from exc
        old_attempts = {item.attempt_id: item for item in previous.attempts}
        new_attempts = {item.attempt_id: item for item in tournament.attempts}
        if not old_attempts.keys() <= new_attempts.keys():
            raise AttemptConflictError("persisted attempts cannot be removed")
        for attempt_id_value, old in old_attempts.items():
            new = new_attempts[attempt_id_value]
            if new == old:
                continue
            if (new.work_id, new.attempt_number, new.attempt_id) != (old.work_id, old.attempt_number, old.attempt_id):
                raise AttemptConflictError("attempt identity cannot change")
            try:
                require_attempt_transition(old.status, new.status)
            except ValueError as exc:
                raise AttemptTransitionError(f"illegal persisted attempt transition: {old.status.value} -> {new.status.value}") from exc
            if old.status is AttemptStatus.RUNNING and new.started_at != old.started_at:
                raise AttemptTransitionError("a running attempt's started_at cannot change")
        for result_id, old in previous.results.items():
            if result_id not in tournament.results:
                raise ResultConflictError("persisted results cannot be removed")
            if tournament.results[result_id] != old:
                raise ResultConflictError(f"persisted result cannot be replaced: {result_id}")
        for name in ("eliminations", "round_results", "selections"):
            old_evidence = getattr(previous, name)
            new_evidence = getattr(tournament, name)
            if new_evidence[:len(old_evidence)] != old_evidence:
                raise EvidenceConflictError(f"persisted {name} cannot be removed or modified")

    def save(self, tournament: PersistedTournament) -> PersistedTournament:
        if not self.exists(tournament.tournament_id):
            raise TournamentStoreError("create must be used before save")
        persisted = self.load(tournament.tournament_id)
        manifest = self.load_manifest(tournament.tournament_id)
        self._validate_replacement(persisted, tournament)
        validate_tournament_against_manifest(tournament, manifest)
        if (tournament.progress_completed, tournament.progress_total) != calculate_progress(manifest, tournament):
            raise TournamentStoreError("persisted progress does not match manifest work")
        self._atomic_json(self.state_path(tournament.tournament_id), tournament_to_dict(tournament))
        return tournament

    def resume_interrupted(self, tournament_id: str, timestamp: datetime, reason: str = "explicit resume") -> PersistedTournament:
        self._validate_timestamp(timestamp, "timestamp")
        previous = self.load(tournament_id)
        target = replace(previous, state=resume_interrupted_state(previous.state), updated_at=timestamp, last_event="tournament resumed", last_reason=reason)
        self._validate_replacement(previous, target, allow_resume=True)
        self._atomic_json(self.state_path(tournament_id), tournament_to_dict(target))
        return target

    @staticmethod
    def _validate_timestamp(timestamp: datetime, name: str) -> None:
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError(f"{name} must be timezone-aware UTC")

    def append_event(self, tournament_id: str, event: str, timestamp: datetime, reason: str | None = None) -> None:
        self.load(tournament_id)
        if not isinstance(event, str) or not event.strip():
            raise ValueError("event must be non-empty")
        self._validate_timestamp(timestamp, "timestamp")
        payload = {"event": event, "reason": reason, "timestamp": timestamp.isoformat()}
        path = self.events_path(tournament_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def has_result(self, tournament_id: str, stable_id: str) -> bool:
        return stable_id in self.load(tournament_id).results

    def record_result(self, tournament_id: str, result: TaskResult, updated_at: datetime) -> TaskResult:
        tournament = self.load(tournament_id)
        self._validate_timestamp(updated_at, "updated_at")
        if updated_at < tournament.updated_at:
            raise TimestampRegressionError("updated_at cannot move backwards")
        existing = tournament.results.get(result.stable_id)
        if existing is not None:
            if existing == result:
                return existing
            raise ResultConflictError(f"different result already exists for {result.stable_id}")
        results = dict(tournament.results)
        results[result.stable_id] = result
        updated = replace(tournament, results=results, updated_at=updated_at, last_event="task result recorded")
        self.save(with_calculated_progress(self.load_manifest(tournament_id), updated))
        return result

    def record_attempt(self, tournament_id: str, attempt: TournamentAttempt, updated_at: datetime) -> TournamentAttempt:
        tournament = self.load(tournament_id)
        self._validate_timestamp(updated_at, "updated_at")
        if updated_at < tournament.updated_at:
            raise TimestampRegressionError("updated_at cannot move backwards")
        if attempt.status is not AttemptStatus.PENDING:
            raise AttemptTransitionError("record_attempt creates PENDING attempts only")
        existing = next((item for item in tournament.attempts if item.attempt_id == attempt.attempt_id), None)
        if existing is not None:
            if existing == attempt:
                return existing
            raise AttemptConflictError(f"different attempt already exists for {attempt.attempt_id}")
        self.save(replace(tournament, attempts=tournament.attempts + (attempt,), updated_at=updated_at, last_event="attempt recorded"))
        return attempt

    def record_elimination(self, tournament_id: str, record: EliminationRecord) -> EliminationRecord:
        tournament = self.load(tournament_id)
        self._validate_timestamp(record.timestamp, "elimination timestamp")
        if record.timestamp < tournament.updated_at:
            raise TimestampRegressionError("elimination timestamp cannot move tournament updated_at backwards")
        existing = next((item for item in tournament.eliminations if item.candidate_id == record.candidate_id), None)
        if existing is not None:
            if existing == record:
                return existing
            raise TournamentStoreError(f"candidate already has an elimination record: {record.candidate_id}")
        updated = eliminate_candidate(tournament, record)
        updated = with_calculated_progress(self.load_manifest(tournament_id), updated)
        self.save(updated)
        return record

    def record_selections(self, tournament_id: str, selections: tuple[TournamentSelection, ...], timestamp: datetime) -> tuple[TournamentSelection, ...]:
        """Append final selection evidence without replacing prior role decisions."""
        tournament = self.load(tournament_id)
        self._validate_timestamp(timestamp, "selection timestamp")
        if timestamp < tournament.updated_at:
            raise TimestampRegressionError("selection timestamp cannot move tournament updated_at backwards")
        if not isinstance(selections, tuple) or any(not isinstance(item, TournamentSelection) for item in selections):
            raise ValueError("selections must be a tuple of TournamentSelection values")
        existing_by_role = {item.role: item for item in tournament.selections}
        additions = []
        for selection in selections:
            existing = existing_by_role.get(selection.role)
            if existing is None:
                existing_by_role[selection.role] = selection
                additions.append(selection)
            elif existing != selection:
                raise EvidenceConflictError(f"selection role already recorded: {selection.role.value}")
        if not additions:
            return tournament.selections
        updated = replace(
            tournament,
            selections=tournament.selections + tuple(additions),
            updated_at=timestamp,
            last_event="selections recorded",
        )
        self.save(updated)
        return updated.selections

    def transition_attempt(self, tournament_id: str, attempt_id_value: str, new_status: AttemptStatus, timestamp: datetime, reason: str | None = None) -> TournamentAttempt:
        tournament = self.load(tournament_id)
        self._validate_timestamp(timestamp, "timestamp")
        if timestamp < tournament.updated_at:
            raise TimestampRegressionError("attempt timestamp cannot move tournament updated_at backwards")
        try:
            current = next(item for item in tournament.attempts if item.attempt_id == attempt_id_value)
        except StopIteration as exc:
            raise TournamentStoreError(f"unknown attempt: {attempt_id_value}") from exc
        if current.status is new_status:
            return current
        try:
            require_attempt_transition(current.status, new_status)
        except ValueError as exc:
            raise AttemptTransitionError(f"illegal attempt transition: {current.status.value} -> {new_status.value}") from exc
        if new_status is AttemptStatus.DEFERRED and (not isinstance(reason, str) or not reason.strip()):
            raise AttemptTransitionError("DEFERRED requires a reason")
        started_at = current.started_at
        finished_at = current.finished_at
        if current.status is AttemptStatus.PENDING and new_status is AttemptStatus.RUNNING:
            started_at = timestamp
        elif current.status is AttemptStatus.PENDING:
            started_at = None if new_status is AttemptStatus.DEFERRED else timestamp
            finished_at = None if new_status is AttemptStatus.DEFERRED else timestamp
        elif current.status is AttemptStatus.RUNNING:
            finished_at = timestamp
        updated_attempt = replace(current, status=new_status, started_at=started_at, finished_at=finished_at, reason=reason)
        attempts = tuple(updated_attempt if item.attempt_id == attempt_id_value else item for item in tournament.attempts)
        updated = replace(tournament, attempts=attempts, updated_at=timestamp, last_event="attempt transitioned", last_reason=reason or new_status.value)
        self.save(updated)
        return updated_attempt
