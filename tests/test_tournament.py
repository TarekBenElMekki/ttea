from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ttea.benchmarking import OllamaClient, OwnedOllamaServer
from ttea.resources import ResourceDecision, ResourceRequest
from ttea.tournament import (
    DIMENSIONS, MAX_CONCURRENCY, PERFORMANCE_METRICS, SCHEMA_VERSION, AttemptStatus, AttemptTransitionError, CandidateState, CandidateTransitionError,
    CandidateVariant, EliminationReason, EliminationRecord,
    ManifestError, ManifestMismatchError, PersistedTournament,
    ProviderAvailability, ProviderMetadata, ProvisioningEligibility,
    EvidenceConflictError, ResultConflictError, SelectionRole, TaskResult, TournamentAttempt,
    TournamentCandidate, TournamentExecutionRequest, TournamentManifest,
    TournamentPhase, TournamentProvider, TournamentResult, TournamentRound,
    TimestampRegressionError, TournamentIntegrityError, TournamentSelection,
    TournamentSerializationError, TournamentState, TournamentTransitionError,
    TournamentStore, TournamentStoreError, TournamentTask,
    TournamentWorkItem, WorkDisposition, aggregate_round, apply_scheduler_decision, attempt_id,
    calculate_progress, composite, eliminate_candidate, next_work,
    ordered_work, recover_stale_attempts, retry_attempt, schedule_resource_decision, select,
    snapshot, task_result_id, tournament_from_dict, tournament_to_dict,
    with_calculated_progress,
)
from ttea.tournament.__main__ import main as tournament_main
from ttea.tournament.reporting import report_payload, text_report, write_report


NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


class FakeTournamentProvider(TournamentProvider):
    def availability(self, candidate):
        return ProviderAvailability(True, "available in fake inventory")

    def metadata(self, candidate):
        return ProviderMetadata(candidate.variant.provider, candidate.variant.model, {"fake": True})

    def provisioning_eligibility(self, candidate):
        return ProvisioningEligibility(True, "fake provisioning would be eligible")

    def resource_request(self, candidate, tournament_round, task):
        return ResourceRequest("tournament-test", candidate.estimated_ram_bytes, candidate.estimated_disk_bytes or 0, candidate.estimated_vram_bytes)

    def build_execution_request(self, candidate, tournament_round, task, repetition):
        return TournamentExecutionRequest(candidate.candidate_id, tournament_round.round_id, task.task_id, repetition, candidate.variant.provider, candidate.variant.model, task.prompt, tournament_round.context_limit, tournament_round.output_limit, task.timeout_seconds)


class TournamentTests(unittest.TestCase):
    def candidate(self, candidate_id="a", *, enabled=True, validation=False):
        return TournamentCandidate(candidate_id, "family", CandidateVariant("fake", f"tag-{candidate_id}", "q4", "1B"), 1, 2, 0, 128, "test", enabled, validation)

    def task(self, task_id="x", repetitions=1):
        return TournamentTask(task_id, "planning", "prompt", "json", "exact", 1.0, 128, 16, 5.0, repetitions)

    def tournament_round(self, round_id="r", phase=TournamentPhase.CAPABILITY, repetitions=1, task_ids=("x",)):
        return TournamentRound(round_id, phase, task_ids, repetitions, 128, 16, "small", "bounded", "weighted", "safe")

    def manifest(self, candidates=None, rounds=None, tasks=None):
        return TournamentManifest(candidates or (self.candidate(),), rounds or (self.tournament_round(),), tasks or (self.task(),))

    def state(self, manifest=None, **changes):
        manifest = manifest or self.manifest()
        base = PersistedTournament(SCHEMA_VERSION, "tour-1", manifest.digest, "manifest.json", TournamentState.CREATED, NOW, NOW, progress_total=len(ordered_work(manifest)), candidate_states={candidate.candidate_id: CandidateState.AVAILABLE for candidate in manifest.candidates})
        return replace(base, **changes)

    def result(self, candidate_id="a", repetition=0, score=0.8):
        return TaskResult(candidate_id, "r", "x", repetition, AttemptStatus.COMPLETED, {"reasoning": score}, {"peak_ram_bytes": 100})

    def complete_dimensions(self, score=0.8):
        return {dimension: score for dimension in DIMENSIONS}

    def create_store(self, root, manifest=None, state=None):
        manifest = manifest or self.manifest()
        state = state or self.state(manifest)
        store = TournamentStore(root)
        store.create(state, manifest)
        return store

    def mutate_state_file(self, store, mutate):
        path = store.state_path("tour-1")
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    # Existing identity, manifest, state, scoring, and selection coverage.
    def test_candidate_identity_is_canonical(self):
        variant = CandidateVariant(" Fake ", " TAG ", "Q4", "1B")
        self.assertEqual(variant.canonical_identity, "fake|tag|q4|1b")

    def test_duplicate_canonical_candidate_identity_rejected(self):
        first = self.candidate("a")
        second = replace(first, candidate_id="b")
        with self.assertRaises(ManifestError):
            TournamentManifest((first, second))

    def test_manifest_rejects_unknown_task_reference(self):
        with self.assertRaises(ManifestError):
            TournamentManifest((self.candidate(),), (self.tournament_round(task_ids=("missing",)),), (self.task(),))

    def test_manifest_digest_is_deterministic(self):
        manifest = self.manifest()
        self.assertEqual(manifest.digest, TournamentManifest.from_dict(manifest.to_dict()).digest)

    def test_task_result_identity_avoids_ambiguous_concatenation(self):
        self.assertNotEqual(task_result_id("ab", "c", "d", 0), task_result_id("a", "bc", "d", 0))

    def test_ineligible_result_has_no_composite(self):
        self.assertIsNone(composite(TournamentResult("a", {"reasoning": 1.0}, {}, False)))

    def test_validation_only_candidate_cannot_be_selected(self):
        result = TournamentResult("a", {"reasoning": 1.0}, {"peak_ram_bytes": 1}, True)
        selections = select((result,), (self.candidate(validation=True),))
        self.assertIsNone(selections[0].candidate_id)

    def test_safety_elimination_cannot_be_selected(self):
        result = TournamentResult("a", {"reasoning": 1.0}, {}, False, EliminationReason.CLEANUP_FAILED)
        self.assertIsNone(select((result,), (self.candidate(),))[0].candidate_id)

    def test_incomplete_perfect_score_has_no_composite(self):
        result = TournamentResult("a", {"reasoning": 1.0}, {}, True)
        self.assertIsNone(composite(result))

    def test_complete_score_profile_is_rankable(self):
        result = TournamentResult("a", self.complete_dimensions(0.8), {}, True)
        self.assertAlmostEqual(composite(result), 0.8)

    def test_zero_weight_dimension_may_be_missing(self):
        weights = {dimension: (1.0 if dimension == "reasoning" else 0.0) for dimension in DIMENSIONS}
        result = TournamentResult("a", {"reasoning": 0.9}, {}, True)
        self.assertEqual(composite(result, weights), 0.9)

    def test_incomplete_candidate_is_not_selected(self):
        incomplete = TournamentResult("a", {"reasoning": 1.0}, {"peak_ram_bytes": 1}, True)
        complete = TournamentResult("b", self.complete_dimensions(0.8), {"peak_ram_bytes": 2}, True)
        selections = select((incomplete, complete), (self.candidate("a"), self.candidate("b")))
        self.assertEqual(selections[0].candidate_id, "b")

    def test_fallback_handles_null_and_missing_ram_metrics(self):
        known = TournamentResult("a", self.complete_dimensions(), {"peak_ram_bytes": 1}, True)
        null = TournamentResult("b", self.complete_dimensions(), {"peak_ram_bytes": None}, True)
        missing = TournamentResult("c", self.complete_dimensions(), {}, True)
        candidates = (self.candidate("a"), self.candidate("b"), self.candidate("c"))
        selections = select((null, missing, known), candidates)
        self.assertEqual(selections[1].candidate_id, "a")

    def test_fallback_prefers_known_lower_ram(self):
        one_gib = TournamentResult("a", self.complete_dimensions(), {"peak_ram_bytes": 1024**3}, True)
        two_gib = TournamentResult("b", self.complete_dimensions(), {"peak_ram_bytes": 2 * 1024**3}, True)
        selections = select((two_gib, one_gib), (self.candidate("a"), self.candidate("b")))
        self.assertEqual(selections[1].candidate_id, "a")

    def test_invalid_standard_metrics_are_rejected(self):
        invalid = {
            "peak_ram_bytes": float("nan"),
            "peak_vram_bytes": float("inf"),
            "cold_load_latency_seconds": -1,
            "gpu_temperature_c": float("inf"),
            "cleanup_reliability": 2,
        }
        for key, value in invalid.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                TournamentResult("a", self.complete_dimensions(), {key: value}, True)

    # Provider and work model.
    def test_provider_contract_can_be_implemented_by_fake(self):
        provider = FakeTournamentProvider()
        manifest = self.manifest()
        self.assertTrue(provider.availability(manifest.candidates[0]).available)
        self.assertIsInstance(provider.metadata(manifest.candidates[0]), ProviderMetadata)
        self.assertTrue(provider.provisioning_eligibility(manifest.candidates[0]).eligible)
        self.assertIsInstance(provider.resource_request(manifest.candidates[0], manifest.rounds[0], manifest.tasks[0]), ResourceRequest)
        self.assertIsInstance(provider.build_execution_request(manifest.candidates[0], manifest.rounds[0], manifest.tasks[0], 0), TournamentExecutionRequest)

    def test_work_item_identity_matches_result(self):
        work = TournamentWorkItem("a", "r", "x", 0)
        self.assertEqual(work.stable_id, self.result().stable_id)

    def test_work_item_validation(self):
        with self.assertRaises(ValueError):
            TournamentWorkItem("", "r", "x", 0)

    def test_work_order_is_deterministic(self):
        manifest = self.manifest((self.candidate("a"), self.candidate("b")), (self.tournament_round(repetitions=2),), (self.task(repetitions=2),))
        work = ordered_work(manifest)
        self.assertEqual([(item.candidate_id, item.repetition) for item in work], [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 0), ("b", 1), ("b", 2), ("b", 3)])

    def test_concurrency_contract_is_one(self):
        self.assertEqual(MAX_CONCURRENCY, 1)

    # Resource scheduling.
    def test_allow_permits_conceptual_progression(self):
        self.assertIs(schedule_resource_decision(ResourceDecision.ALLOW, "safe").disposition, WorkDisposition.PROCEED)

    def test_delay_defers_without_elimination(self):
        decision = schedule_resource_decision(ResourceDecision.DELAY, "temporarily warm")
        self.assertIs(decision.disposition, WorkDisposition.DEFER)
        self.assertIsNone(decision.elimination_reason)
        state = apply_scheduler_decision(self.state(), TournamentWorkItem("a", "r", "x", 0), decision, NOW + timedelta(seconds=1))
        self.assertIs(state.state, TournamentState.WAITING_FOR_RESOURCES)
        self.assertIs(state.attempts[-1].status, AttemptStatus.DEFERRED)
        self.assertEqual(state.eliminations, ())

    def test_deny_can_produce_elimination(self):
        decision = schedule_resource_decision(ResourceDecision.DENY, "incompatible")
        self.assertIs(decision.disposition, WorkDisposition.ELIMINATE)
        self.assertIs(decision.elimination_reason, EliminationReason.RESOURCE_DENIED)
        state = apply_scheduler_decision(self.state(), TournamentWorkItem("a", "r", "x", 0), decision, NOW + timedelta(seconds=1))
        self.assertIs(state.candidate_states["a"], CandidateState.ELIMINATED)

    def test_deferred_work_remains_next(self):
        state = self.state(attempts=(TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1),))
        deferred = replace(state.attempts[0], status=AttemptStatus.DEFERRED, reason="later")
        self.assertEqual(next_work(self.manifest(), replace(state, attempts=(deferred,))).stable_id, deferred.work_id)

    # Persistence and serialization.
    def test_persisted_round_trip(self):
        state = self.state()
        self.assertEqual(tournament_from_dict(tournament_to_dict(state)), state)

    def test_malformed_enum_rejected(self):
        value = tournament_to_dict(self.state())
        value["state"] = "BROKEN"
        with self.assertRaises(TournamentSerializationError):
            tournament_from_dict(value)

    def test_malformed_timestamp_rejected(self):
        value = tournament_to_dict(self.state())
        value["created_at"] = "not-a-time"
        with self.assertRaises(TournamentSerializationError):
            tournament_from_dict(value)

    def test_non_utc_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.state(), updated_at=datetime.now())

    def test_bad_schema_version_rejected(self):
        value = tournament_to_dict(self.state())
        value["schema_version"] = 999
        with self.assertRaises(TournamentSerializationError):
            tournament_from_dict(value)

    def test_impossible_progress_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.state(), progress_completed=2, progress_total=1)

    def test_persisted_mappings_are_frozen_and_copied(self):
        source = {"a": CandidateState.AVAILABLE}
        state = replace(self.state(), candidate_states=source)
        source["a"] = CandidateState.ELIMINATED
        self.assertIs(state.candidate_states["a"], CandidateState.AVAILABLE)
        with self.assertRaises(TypeError):
            state.candidate_states["a"] = CandidateState.RUNNING

    def test_frozen_model_rejects_assignment(self):
        with self.assertRaises(FrozenInstanceError):
            self.state().state = TournamentState.RUNNING

    def test_terminal_states_cannot_return_to_running_via_save(self):
        for terminal in (TournamentState.COMPLETED, TournamentState.FAILED, TournamentState.CANCELLED):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as directory:
                store = self.create_store(directory, state=self.state(state=terminal))
                with self.assertRaises(TournamentTransitionError):
                    store.save(replace(store.load("tour-1"), state=TournamentState.RUNNING, updated_at=NOW + timedelta(seconds=1)))

    def test_interrupted_generic_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory, state=self.state(state=TournamentState.INTERRUPTED))
            with self.assertRaises(TournamentTransitionError):
                store.save(replace(store.load("tour-1"), state=TournamentState.RUNNING, updated_at=NOW + timedelta(seconds=1)))

    def test_explicit_interrupted_resume_persists_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory, state=self.state(state=TournamentState.INTERRUPTED))
            resumed = store.resume_interrupted("tour-1", NOW + timedelta(seconds=1), "operator resume")
            self.assertIs(resumed.state, TournamentState.PAUSED)
            self.assertIs(store.load("tour-1").state, TournamentState.PAUSED)

    def test_legal_and_unchanged_state_saves_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            running = store.save(replace(store.load("tour-1"), state=TournamentState.RUNNING, updated_at=NOW + timedelta(seconds=1)))
            store.save(replace(running, last_reason="checkpoint", updated_at=NOW + timedelta(seconds=2)))
            self.assertEqual(store.load("tour-1").last_reason, "checkpoint")

    def test_scheduler_helper_rejects_illegal_terminal_transition(self):
        decision = schedule_resource_decision(ResourceDecision.ALLOW, "safe")
        with self.assertRaises(ValueError):
            apply_scheduler_decision(self.state(state=TournamentState.COMPLETED), TournamentWorkItem("a", "r", "x", 0), decision, NOW + timedelta(seconds=1))

    def test_candidate_state_transitions_are_enforced_on_save(self):
        illegal_pairs = (
            (CandidateState.ELIMINATED, CandidateState.AVAILABLE),
            (CandidateState.ELIMINATED, CandidateState.RUNNING),
            (CandidateState.SELECTED, CandidateState.RUNNING),
            (CandidateState.AVAILABLE, CandidateState.FINALIST),
            (CandidateState.PENDING, CandidateState.SELECTED),
        )
        for previous_state, replacement_state in illegal_pairs:
            with self.subTest(previous=previous_state, replacement=replacement_state), tempfile.TemporaryDirectory() as directory:
                initial = with_calculated_progress(self.manifest(), self.state(candidate_states={"a": previous_state}))
                store = self.create_store(directory, state=initial)
                with self.assertRaises(CandidateTransitionError):
                    store.save(replace(store.load("tour-1"), candidate_states={"a": replacement_state}, updated_at=NOW + timedelta(seconds=1)))

    def test_candidate_available_to_running_is_legal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            updated = store.save(replace(store.load("tour-1"), candidate_states={"a": CandidateState.RUNNING}, updated_at=NOW + timedelta(seconds=1)))
            self.assertIs(updated.candidate_states["a"], CandidateState.RUNNING)

    def test_candidate_state_cannot_disappear_or_be_inserted(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            current = store.load("tour-1")
            with self.assertRaises(CandidateTransitionError):
                store.save(replace(current, candidate_states={}, updated_at=NOW + timedelta(seconds=1)))
            with self.assertRaises(CandidateTransitionError):
                store.save(replace(current, candidate_states={"a": CandidateState.AVAILABLE, "new": CandidateState.PENDING}, updated_at=NOW + timedelta(seconds=1)))

    # Store and manifest compatibility.
    def test_create_writes_state_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            self.assertTrue(store.state_path("tour-1").is_file())
            self.assertTrue(store.manifest_snapshot_path("tour-1").is_file())

    def test_save_uses_atomic_replacement_without_leftovers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.save(replace(store.load("tour-1"), state=TournamentState.RUNNING))
            self.assertIs(store.load("tour-1").state, TournamentState.RUNNING)
            self.assertEqual(list(store.tournament_path("tour-1").glob("*.tmp")), [])

    def test_store_list_and_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            self.assertTrue(store.exists("tour-1"))
            self.assertEqual(store.list()[0].tournament_id, "tour-1")

    def test_save_rejects_backward_timestamp_and_changed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = self.state(updated_at=NOW + timedelta(seconds=2))
            store = self.create_store(directory, state=initial)
            with self.assertRaises(TimestampRegressionError):
                store.save(replace(initial, updated_at=NOW + timedelta(seconds=1)))
            with self.assertRaises(ManifestMismatchError):
                store.save(replace(initial, manifest_digest="0" * 64, updated_at=NOW + timedelta(seconds=3)))

    def test_result_and_attempt_recording_reject_backward_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = self.state(updated_at=NOW + timedelta(seconds=2))
            store = self.create_store(directory, state=initial)
            with self.assertRaises(TimestampRegressionError):
                store.record_result("tour-1", self.result(), NOW + timedelta(seconds=1))
            attempt = TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1)
            with self.assertRaises(TimestampRegressionError):
                store.record_attempt("tour-1", attempt, NOW + timedelta(seconds=1))

    def test_event_append(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.append_event("tour-1", "created", NOW, "test")
            event = json.loads(store.events_path("tour-1").read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "created")

    def test_result_existence_and_identical_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            result = self.result()
            store.record_result("tour-1", result, NOW + timedelta(seconds=1))
            self.assertTrue(store.has_result("tour-1", result.stable_id))
            self.assertEqual(store.load("tour-1").progress_completed, 1)
            self.assertEqual(store.record_result("tour-1", result, NOW + timedelta(seconds=2)), result)

    def test_conflicting_result_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.record_result("tour-1", self.result(), NOW + timedelta(seconds=1))
            with self.assertRaises(ResultConflictError):
                store.record_result("tour-1", self.result(score=0.1), NOW + timedelta(seconds=2))

    def test_corrupted_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.state_path("tour-1").write_text("{", encoding="utf-8")
            with self.assertRaises(TournamentStoreError):
                store.load("tour-1")

    def test_duplicate_json_result_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            text = store.state_path("tour-1").read_text(encoding="utf-8")
            text = text.replace('"results": {}', '"results": {}, "results": {}')
            store.state_path("tour-1").write_text(text, encoding="utf-8")
            with self.assertRaises(TournamentStoreError):
                store.load("tour-1")

    def test_persisted_snapshot_digest_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.verify_manifest("tour-1", self.manifest())

    def test_changed_current_manifest_raises_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            with self.assertRaises(ManifestMismatchError):
                store.verify_manifest("tour-1", self.manifest((self.candidate("b"),)))

    def test_tampered_persisted_manifest_raises_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            value = json.loads(store.manifest_snapshot_path("tour-1").read_text(encoding="utf-8"))
            value["candidates"][0]["purpose"] = "tampered"
            store.manifest_snapshot_path("tour-1").write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ManifestMismatchError):
                store.load("tour-1")

    def test_unknown_candidate_state_in_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            self.mutate_state_file(store, lambda value: value["candidate_states"].update({"unknown": "AVAILABLE"}))
            with self.assertRaises(TournamentIntegrityError):
                store.load("tour-1")

    def test_unknown_result_candidate_round_and_task_are_rejected(self):
        cases = (("unknown", "r", "x"), ("a", "unknown", "x"), ("a", "r", "unknown"))
        for candidate_id, round_id, task_id in cases:
            with self.subTest(candidate=candidate_id, round=round_id, task=task_id), tempfile.TemporaryDirectory() as directory:
                store = self.create_store(directory)
                stable_id = task_result_id(candidate_id, round_id, task_id, 0)
                payload = {"candidate_id": candidate_id, "round_id": round_id, "task_id": task_id, "repetition": 0, "status": "COMPLETED", "dimensions": {}, "metrics": {}, "elimination": None}
                self.mutate_state_file(store, lambda value: value["results"].update({stable_id: payload}))
                with self.assertRaises(TournamentIntegrityError):
                    store.load("tour-1")

    def test_result_task_not_in_round_is_rejected(self):
        manifest = self.manifest(tasks=(self.task("x"), self.task("y")))
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory, manifest)
            stable_id = task_result_id("a", "r", "y", 0)
            payload = {"candidate_id": "a", "round_id": "r", "task_id": "y", "repetition": 0, "status": "COMPLETED", "dimensions": {}, "metrics": {}, "elimination": None}
            self.mutate_state_file(store, lambda value: value["results"].update({stable_id: payload}))
            with self.assertRaises(TournamentIntegrityError):
                store.load("tour-1")

    def test_arbitrary_attempt_work_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            attempt = {"attempt_id": attempt_id("arbitrary", 1), "work_id": "arbitrary", "attempt_number": 1, "status": "PENDING", "started_at": None, "finished_at": None, "reason": None}
            self.mutate_state_file(store, lambda value: value["attempts"].append(attempt))
            with self.assertRaises(TournamentIntegrityError):
                store.load("tour-1")

    def test_invalid_aggregate_reference_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            aggregate = {"candidate_id": "a", "round_id": "r", "task_result_ids": ["missing"], "status": "COMPLETED", "dimensions": {}, "metrics": {}, "eligible": True, "elimination": None}
            self.mutate_state_file(store, lambda value: value["round_results"].append(aggregate))
            with self.assertRaises(TournamentIntegrityError):
                store.load("tour-1")

    def test_validation_only_selection_is_rejected_on_load(self):
        validation = self.candidate("validation", validation=True)
        manifest = self.manifest((self.candidate(), validation))
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory, manifest)
            selection = {"role": "PRIMARY", "candidate_id": "validation", "reason": "tampered"}
            self.mutate_state_file(store, lambda value: value["selections"].append(selection))
            with self.assertRaises(TournamentIntegrityError):
                store.load("tour-1")

    # Attempts and recovery.
    def test_attempt_identity_is_deterministic(self):
        self.assertEqual(attempt_id("work", 1), TournamentAttempt.new("work", 1).attempt_id)

    def test_running_attempt_becomes_interrupted(self):
        attempt = replace(TournamentAttempt.new("work", 1), status=AttemptStatus.RUNNING, started_at=NOW)
        recovered = recover_stale_attempts(self.state(state=TournamentState.RUNNING, attempts=(attempt,)), NOW + timedelta(seconds=1))
        self.assertIs(recovered.attempts[0].status, AttemptStatus.INTERRUPTED)
        self.assertEqual(recovered.results, {})

    def test_completed_attempt_stays_completed(self):
        attempt = TournamentAttempt(attempt_id("work", 1), "work", 1, AttemptStatus.COMPLETED, NOW, NOW)
        self.assertEqual(recover_stale_attempts(self.state(attempts=(attempt,)), NOW + timedelta(seconds=1)).attempts[0], attempt)

    def test_retry_creates_new_attempt(self):
        first = replace(TournamentAttempt.new("work", 1), status=AttemptStatus.DEFERRED, reason="later")
        second = retry_attempt(first)
        self.assertNotEqual(first.attempt_id, second.attempt_id)

    def test_completed_attempt_cannot_be_retried(self):
        completed = TournamentAttempt(attempt_id("work", 1), "work", 1, AttemptStatus.COMPLETED, NOW, NOW)
        with self.assertRaises(ValueError):
            retry_attempt(completed)

    def test_attempt_survives_store_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            attempt = TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1)
            store.record_attempt("tour-1", attempt, NOW + timedelta(seconds=1))
            self.assertEqual(TournamentStore(directory).load("tour-1").attempts, (attempt,))

    def test_attempt_pending_running_completed_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            pending = TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1)
            store.record_attempt("tour-1", pending, NOW + timedelta(seconds=1))
            running = store.transition_attempt("tour-1", pending.attempt_id, AttemptStatus.RUNNING, NOW + timedelta(seconds=2))
            self.assertEqual(running.started_at, NOW + timedelta(seconds=2))
            completed = store.transition_attempt("tour-1", pending.attempt_id, AttemptStatus.COMPLETED, NOW + timedelta(seconds=3))
            self.assertEqual(completed.started_at, running.started_at)
            self.assertEqual(completed.finished_at, NOW + timedelta(seconds=3))

    def test_running_attempt_can_fail_or_be_interrupted(self):
        for terminal in (AttemptStatus.FAILED, AttemptStatus.INTERRUPTED):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as directory:
                store = self.create_store(directory)
                pending = TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1)
                store.record_attempt("tour-1", pending, NOW + timedelta(seconds=1))
                store.transition_attempt("tour-1", pending.attempt_id, AttemptStatus.RUNNING, NOW + timedelta(seconds=2))
                result = store.transition_attempt("tour-1", pending.attempt_id, terminal, NOW + timedelta(seconds=3), "terminal")
                self.assertEqual(result.finished_at, NOW + timedelta(seconds=3))

    def test_pending_attempt_can_be_deferred_without_execution_time(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            pending = TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1)
            store.record_attempt("tour-1", pending, NOW + timedelta(seconds=1))
            deferred = store.transition_attempt("tour-1", pending.attempt_id, AttemptStatus.DEFERRED, NOW + timedelta(seconds=2), "resources busy")
            self.assertIsNone(deferred.started_at)
            self.assertIsNone(deferred.finished_at)
            with self.assertRaises(AttemptTransitionError):
                store.transition_attempt("tour-1", pending.attempt_id, AttemptStatus.RUNNING, NOW + timedelta(seconds=3))

    def test_running_attempt_requires_started_at(self):
        with self.assertRaises(ValueError):
            TournamentAttempt(attempt_id("work", 1), "work", 1, AttemptStatus.RUNNING)

    def test_deferred_attempt_requires_reason_and_no_timestamps(self):
        with self.assertRaises(ValueError):
            TournamentAttempt(attempt_id("work", 1), "work", 1, AttemptStatus.DEFERRED)

    # Resume and progress.
    def test_completed_work_is_skipped(self):
        manifest = self.manifest(rounds=(self.tournament_round(repetitions=2),))
        result = self.result(repetition=0)
        state = self.state(manifest, results={result.stable_id: result})
        self.assertEqual(next_work(manifest, state).repetition, 1)

    def test_eliminated_candidate_is_skipped(self):
        manifest = self.manifest((self.candidate("a"), self.candidate("b")))
        state = self.state(manifest, candidate_states={"a": CandidateState.ELIMINATED, "b": CandidateState.AVAILABLE})
        self.assertEqual(next_work(manifest, state).candidate_id, "b")

    def test_disabled_candidate_is_skipped(self):
        manifest = self.manifest((self.candidate("a", enabled=False), self.candidate("b")))
        self.assertEqual(next_work(manifest, self.state(manifest)).candidate_id, "b")

    def test_available_state_survives_serialization(self):
        state = tournament_from_dict(tournament_to_dict(self.state()))
        self.assertIs(state.candidate_states["a"], CandidateState.AVAILABLE)

    def test_next_work_is_deterministic_after_restart(self):
        manifest = self.manifest(rounds=(self.tournament_round(repetitions=2),))
        state = self.state(manifest)
        self.assertEqual(next_work(manifest, state), next_work(manifest, tournament_from_dict(tournament_to_dict(state))))

    def test_completed_work_increments_progress(self):
        result = self.result()
        state = replace(self.state(), results={result.stable_id: result})
        self.assertEqual(calculate_progress(self.manifest(), state), (1, 1))

    def test_temporary_delay_does_not_increment_progress(self):
        deferred = replace(TournamentAttempt.new(TournamentWorkItem("a", "r", "x", 0).stable_id, 1), status=AttemptStatus.DEFERRED, reason="later")
        self.assertEqual(calculate_progress(self.manifest(), self.state(attempts=(deferred,))), (0, 1))

    def test_eliminated_candidate_does_not_leave_unfinished_work(self):
        state = replace(self.state(), candidate_states={"a": CandidateState.ELIMINATED})
        self.assertEqual(calculate_progress(self.manifest(), state), (0, 0))

    def test_progress_is_deterministic_after_round_trip(self):
        state = with_calculated_progress(self.manifest(), self.state())
        restored = tournament_from_dict(tournament_to_dict(state))
        self.assertEqual(calculate_progress(self.manifest(), state), calculate_progress(self.manifest(), restored))

    def test_completed_evidence_survives_later_elimination(self):
        manifest = self.manifest(rounds=(self.tournament_round(repetitions=2),))
        result = self.result(repetition=0)
        before = self.state(manifest, results={result.stable_id: result})
        record = EliminationRecord("a", EliminationReason.THERMAL_ABORT, NOW + timedelta(seconds=1), "unsafe")
        after = eliminate_candidate(before, record)
        self.assertEqual(calculate_progress(manifest, before), (1, 2))
        self.assertEqual(calculate_progress(manifest, after), (1, 1))
        self.assertEqual(calculate_progress(manifest, tournament_from_dict(tournament_to_dict(after))), (1, 1))

    def test_task_result_accepts_only_completed_or_failed(self):
        for status in (AttemptStatus.PENDING, AttemptStatus.RUNNING, AttemptStatus.DEFERRED, AttemptStatus.INTERRUPTED):
            with self.subTest(status=status), self.assertRaises(ValueError):
                TaskResult("a", "r", "x", 0, status)
        self.assertIs(TaskResult("a", "r", "x", 0, AttemptStatus.COMPLETED).status, AttemptStatus.COMPLETED)
        self.assertIs(TaskResult("a", "r", "x", 0, AttemptStatus.FAILED).status, AttemptStatus.FAILED)

    def test_failed_task_result_is_terminal_for_planning(self):
        failed = TaskResult("a", "r", "x", 0, AttemptStatus.FAILED)
        state = replace(self.state(), results={failed.stable_id: failed})
        self.assertIsNone(next_work(self.manifest(), state))
        self.assertEqual(calculate_progress(self.manifest(), state), (1, 1))

    def test_non_finite_and_arbitrary_metrics_are_rejected(self):
        for invalid in (float("nan"), float("inf"), float("-inf"), object()):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                TaskResult("a", "r", "x", 0, AttemptStatus.COMPLETED, metrics={"value": invalid})

    def test_non_string_metric_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            TaskResult("a", "r", "x", 0, AttemptStatus.COMPLETED, metrics={"nested": {1: "bad"}})
        with self.assertRaises(ValueError):
            TaskResult("a", "r", "x", 0, AttemptStatus.COMPLETED, metrics={1: "bad"})

    def test_nested_metrics_are_immutable_copies_and_round_trip(self):
        source = {"samples": [1, {"temperature": 42.5}], "ok": True}
        result = TaskResult("a", "r", "x", 0, AttemptStatus.COMPLETED, metrics=source)
        source["samples"][1]["temperature"] = 99
        self.assertEqual(result.metrics["samples"][1]["temperature"], 42.5)
        with self.assertRaises(TypeError):
            result.metrics["samples"][1]["temperature"] = 5
        state = replace(self.state(), results={result.stable_id: result}, progress_completed=1)
        self.assertEqual(tournament_from_dict(tournament_to_dict(state)), state)

    # Aggregation and elimination.
    def test_round_aggregate_validates_and_preserves_result_references(self):
        result = self.result()
        aggregate = aggregate_round("a", "r", (result.stable_id,), {result.stable_id: result}, eligible=True)
        self.assertEqual(aggregate.task_result_ids, (result.stable_id,))
        self.assertEqual(aggregate.dimensions["reasoning"], 0.8)

    def test_round_aggregate_rejects_unknown_result(self):
        with self.assertRaises(ManifestError):
            aggregate_round("a", "r", ("missing",), {}, eligible=True)

    def test_safety_elimination_is_persistable_and_makes_candidate_ineligible(self):
        record = EliminationRecord("a", EliminationReason.THERMAL_ABORT, NOW + timedelta(seconds=1), "temperature limit", "r", "x", "work")
        state = eliminate_candidate(self.state(), record)
        restored = tournament_from_dict(tournament_to_dict(state))
        self.assertIs(restored.candidate_states["a"], CandidateState.ELIMINATED)
        self.assertIsNone(next_work(self.manifest(), restored))

    def test_store_records_elimination_with_monotonic_progress(self):
        manifest = self.manifest(rounds=(self.tournament_round(repetitions=2),))
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory, manifest)
            store.record_result("tour-1", self.result(repetition=0), NOW + timedelta(seconds=1))
            record = EliminationRecord("a", EliminationReason.THERMAL_ABORT, NOW + timedelta(seconds=2), "unsafe")
            store.record_elimination("tour-1", record)
            state = store.load("tour-1")
            self.assertEqual((state.progress_completed, state.progress_total), (1, 1))
            self.assertEqual(state.eliminations, (record,))

    def test_elimination_evidence_cannot_be_removed_or_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            record = EliminationRecord("a", EliminationReason.THERMAL_ABORT, NOW + timedelta(seconds=1), "unsafe")
            store.record_elimination("tour-1", record)
            state = store.load("tour-1")
            with self.assertRaises(EvidenceConflictError):
                store.save(replace(state, eliminations=(), updated_at=NOW + timedelta(seconds=2)))
            altered = replace(record, explanation="rewritten")
            with self.assertRaises(EvidenceConflictError):
                store.save(replace(state, eliminations=(altered,), updated_at=NOW + timedelta(seconds=2)))

    def test_elimination_requires_eliminated_candidate_state(self):
        record = EliminationRecord("a", EliminationReason.THERMAL_ABORT, NOW + timedelta(seconds=1), "unsafe")
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            invalid = replace(store.load("tour-1"), eliminations=(record,), updated_at=NOW + timedelta(seconds=1))
            with self.assertRaises(TournamentIntegrityError):
                store.save(invalid)

    def test_round_and_selection_evidence_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            result = self.result()
            store.record_result("tour-1", result, NOW + timedelta(seconds=1))
            aggregate = aggregate_round("a", "r", (result.stable_id,), {result.stable_id: result}, eligible=True)
            with_aggregate = store.save(replace(store.load("tour-1"), round_results=(aggregate,), updated_at=NOW + timedelta(seconds=2)))
            with self.assertRaises(EvidenceConflictError):
                store.save(replace(with_aggregate, round_results=(), updated_at=NOW + timedelta(seconds=3)))
            altered = replace(aggregate, eligible=False)
            with self.assertRaises(EvidenceConflictError):
                store.save(replace(with_aggregate, round_results=(altered,), updated_at=NOW + timedelta(seconds=3)))
            selection = TournamentSelection(SelectionRole.PRIMARY, "a", "chosen")
            with_selection = store.record_selections("tour-1", (selection,), NOW + timedelta(seconds=3))
            state = store.load("tour-1")
            self.assertEqual(with_selection, (selection,))
            with self.assertRaises(EvidenceConflictError):
                store.save(replace(state, selections=(), updated_at=NOW + timedelta(seconds=4)))
            changed = TournamentSelection(SelectionRole.PRIMARY, None, "changed")
            with self.assertRaises(EvidenceConflictError):
                store.save(replace(state, selections=(changed,), updated_at=NOW + timedelta(seconds=4)))

    def test_eliminated_candidate_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.record_elimination("tour-1", EliminationRecord("a", EliminationReason.RESOURCE_DENIED, NOW + timedelta(seconds=1), "unsafe"))
            with self.assertRaises(TournamentIntegrityError):
                store.record_selections("tour-1", (TournamentSelection(SelectionRole.PRIMARY, "a", "invalid"),), NOW + timedelta(seconds=2))

    def test_delay_creates_no_elimination(self):
        self.assertEqual(self.state().eliminations, ())
        self.assertIsNone(schedule_resource_decision(ResourceDecision.DELAY, "later").elimination_reason)

    def test_persisted_elimination_excludes_selection(self):
        result = TournamentResult("a", {"reasoning": 1.0}, {}, True)
        selections = select((result,), (self.candidate(),), eliminated_candidate_ids=("a",))
        self.assertIsNone(selections[0].candidate_id)

    # Reporting and snapshot.
    def test_json_report_contains_null_missing_metrics_and_dimensions(self):
        result = self.result()
        aggregate = aggregate_round("a", "r", (result.stable_id,), {result.stable_id: result}, eligible=True)
        payload = report_payload(replace(self.state(), round_results=(aggregate,)), self.manifest())
        self.assertIsNone(payload["round_aggregates"][0]["metrics"]["peak_vram_bytes"])
        self.assertEqual(payload["round_aggregates"][0]["dimensions"]["reasoning"], 0.8)
        self.assertEqual(set(payload["round_aggregates"][0]["metrics"]), set(PERFORMANCE_METRICS))

    def test_report_represents_eliminations_and_selections(self):
        record = EliminationRecord("a", EliminationReason.RESOURCE_DENIED, NOW, "unsafe")
        selection = TournamentSelection(SelectionRole.PRIMARY, None, "no eligible candidate")
        payload = report_payload(replace(self.state(), eliminations=(record,), selections=(selection,)))
        self.assertEqual(payload["eliminations"][0]["reason"], "RESOURCE_DENIED")
        self.assertEqual(payload["selections"][0]["role"], "PRIMARY")

    def test_text_report_is_useful(self):
        text = text_report(self.state(), self.manifest())
        self.assertIn("T/TEA Tournament", text)
        self.assertIn("Selections:\nPrimary:", text)
        self.assertNotIn("â€”", text)

    def test_incomplete_aggregate_is_not_misleadingly_ranked(self):
        result = self.result()
        aggregate = aggregate_round("a", "r", (result.stable_id,), {result.stable_id: result}, eligible=True)
        payload = report_payload(replace(self.state(), round_results=(aggregate,)), self.manifest())
        self.assertIsNone(payload["round_aggregates"][0]["composite"])
        self.assertLess(payload["round_aggregates"][0]["score_coverage"], 1)
        self.assertEqual(payload["ranking"], [])

    def test_report_file_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            json_path, text_path = write_report(store, store.load("tour-1"), self.manifest())
            self.assertTrue(Path(json_path).is_file())
            self.assertTrue(Path(text_path).is_file())

    def test_snapshot_uses_latest_aggregate_and_elimination(self):
        result = self.result()
        aggregate = aggregate_round("a", "r", (result.stable_id,), {result.stable_id: result}, eligible=True)
        record = EliminationRecord("a", EliminationReason.RUNTIME_FAILED, NOW, "failed")
        value = snapshot(replace(self.state(), round_results=(aggregate,), eliminations=(record,)))
        self.assertEqual(value.latest_scores["reasoning"], 0.8)
        self.assertEqual(value.latest_elimination, record)

    # Read-only CLI.
    def run_cli(self, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            code = tournament_main(arguments)
        return code, output.getvalue()

    def test_cli_list_sees_stored_tournament(self):
        with tempfile.TemporaryDirectory() as directory:
            self.create_store(directory)
            code, output = self.run_cli(("--root", directory, "list"))
            self.assertEqual(code, 0)
            self.assertIn("tour-1", output)

    def test_cli_status_sees_stored_tournament(self):
        with tempfile.TemporaryDirectory() as directory:
            self.create_store(directory)
            code, output = self.run_cli(("--root", directory, "status", "tour-1"))
            self.assertEqual(code, 0)
            self.assertIn("Progress: 0/1", output)

    def test_cli_report_renders_stored_tournament(self):
        with tempfile.TemporaryDirectory() as directory:
            self.create_store(directory)
            code, output = self.run_cli(("--root", directory, "report", "tour-1"))
            self.assertEqual(code, 0)
            self.assertIn("T/TEA Tournament", output)

    def test_cli_manifest_sees_persisted_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest()
            self.create_store(directory, manifest)
            code, output = self.run_cli(("--root", directory, "manifest", "tour-1"))
            self.assertEqual(code, 0)
            self.assertIn(manifest.digest, output)

    def test_cli_start_refuses(self):
        code, output = self.run_cli(("start",))
        self.assertEqual(code, 2)
        self.assertIn("execution not enabled", output)

    def test_framework_inspection_is_inert(self):
        with tempfile.TemporaryDirectory() as directory:
            self.create_store(directory)
            with patch.object(OllamaClient, "pull", side_effect=AssertionError("pull called")) as pull, patch.object(OllamaClient, "generate", side_effect=AssertionError("generate called")) as generate, patch.object(OwnedOllamaServer, "start", side_effect=AssertionError("server called")) as start, patch.object(OwnedOllamaServer, "shutdown", side_effect=AssertionError("termination called")) as shutdown:
                for command in ("list", "status", "report", "manifest"):
                    arguments = ("--root", directory, command) if command == "list" else ("--root", directory, command, "tour-1")
                    self.assertEqual(self.run_cli(arguments)[0], 0)
                pull.assert_not_called()
                generate.assert_not_called()
                start.assert_not_called()
                shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
