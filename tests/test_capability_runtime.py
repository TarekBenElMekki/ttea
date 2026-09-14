from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from ttea.benchmarking import BenchmarkOutcome, BenchmarkResult, OllamaGenerationResult
from ttea.capability import CAPABILITY_FIXTURES
from ttea.resources import ResourceDecision
from ttea.tournament.capability import (
    ROUND2_CANDIDATES,
    CapabilityTerminalState,
    CapabilityWorkId,
    CapabilityWorkStore,
    RawCapabilityResponse,
    round2_manifest_from_source,
)
from ttea.tournament.capability_runtime import (
    CAPABILITY_MAX_TECHNICAL_ATTEMPTS,
    CapabilityRuntimeEngine,
    capability_execution_request,
    render_capability_prompt,
)
from ttea.tournament.catalog import build_survival_manifest
from ttea.tournament.ollama_provider import SurvivalExecutionResult


GOOD = {
    "cap.protocol.basic": {"mode":"ACT","plan":[{"subgoal":"prepare","action":"prepare_output","depends_on":[]}]},
    "cap.constraint.preserve_input": {"mode":"ACT","plan":[{"subgoal":"prepare","action":"prepare_output","depends_on":[]}]},
    "cap.policy.guard_boundary": {"mode":"ACT","plan":[{"subgoal":"obtain_approval","action":"request_approval","depends_on":[]}]},
    "cap.state.no_action": {"mode":"NO_ACTION","plan":[]},
    "cap.state.precondition_choice": {"mode":"ACT","plan":[{"subgoal":"commit","action":"commit_output","depends_on":[]}]},
    "cap.plan.chain": {"mode":"ACT","plan":[
        {"subgoal":"collect","action":"inspect_input","depends_on":[]},
        {"subgoal":"validate","action":"validate_item","depends_on":["collect"]},
        {"subgoal":"prepare","action":"prepare_output","depends_on":["validate"]},
        {"subgoal":"commit","action":"commit_output","depends_on":["prepare"]},
    ]},
    "cap.plan.fork_join": {"mode":"ACT","plan":[
        {"subgoal":"collect_input","action":"inspect_input","depends_on":[]},
        {"subgoal":"collect_policy","action":"inspect_policy","depends_on":[]},
        {"subgoal":"validate","action":"validate_item","depends_on":["collect_input","collect_policy"]},
        {"subgoal":"prepare","action":"prepare_output","depends_on":["validate"]},
    ]},
    "cap.replan.failed_alternative": {"mode":"ACT","plan":[{"subgoal":"prepare","action":"use_alternative","depends_on":[]}]},
    "cap.replan.state_delta": {"mode":"ACT","plan":[{"subgoal":"commit","action":"commit_output","depends_on":[]}]},
}


def encoded(fixture_id):
    return json.dumps(GOOD[fixture_id], separators=(",", ":"))


class FakeResources:
    def __init__(self, decisions=()):
        self.decisions = list(decisions)
        self.calls = 0

    def assess(self, request):
        self.calls += 1
        decision = self.decisions.pop(0) if self.decisions else ResourceDecision.ALLOW
        return SimpleNamespace(
            decision=decision,
            reasons=(f"resource {decision.value.lower()}",),
            selected_gpu_index=None,
        )


class FakeProvider:
    def __init__(self, *, responses=None, outcomes=()):
        self.responses = responses or {}
        self.outcomes = list(outcomes)
        self.calls = []

    def execute_prompt(self, candidate, request, resource_request):
        self.calls.append((candidate.candidate_id, request.task_id))
        outcome = self.outcomes.pop(0) if self.outcomes else BenchmarkOutcome.COMPLETED
        response = self.responses.get(request.task_id, encoded(request.task_id))
        generated = None
        if response is not None:
            generated = OllamaGenerationResult(
                candidate.variant.model,
                response,
                None, None, None, None, None, None,
                "stop",
            )
        now = datetime.now(timezone.utc)
        benchmark = BenchmarkResult(
            outcome,
            ("fake benchmark",),
            now,
            now,
            101,
            generated,
            True,
            None,
            1.0,
        )
        return SurvivalExecutionResult(
            benchmark,
            {},
            True,
            True,
            True,
            202,
            2.0,
        )


class NoInferenceProvider:
    def execute_prompt(self, *args, **kwargs):
        raise AssertionError("persisted raw response must not be regenerated")


class CapabilityRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.source = build_survival_manifest()
        self.manifest = round2_manifest_from_source(self.source)

    def engine(self, directory, provider=None, resources=None, tournament_id="cap-runtime"):
        store = CapabilityWorkStore(Path(directory))
        provider = provider or FakeProvider()
        resources = resources or FakeResources()
        engine = CapabilityRuntimeEngine(
            tournament_id,
            store,
            self.manifest,
            provider,
            resources,
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
        )
        return engine, store, provider, resources

    def test_prompt_is_deterministic_and_contains_no_oracle_or_scoring_fields(self):
        fixture = next(item for item in CAPABILITY_FIXTURES if item.id == "cap.plan.chain")
        first = render_capability_prompt(fixture)
        second = render_capability_prompt(fixture)
        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertEqual(parsed["mission"], fixture.input.mission)
        self.assertEqual(parsed["subgoal_ids"], [item.id for item in fixture.subgoals])
        for forbidden in (
            "oracle",
            "required_subgoal_ids",
            "dependency_edges",
            "required_order",
            "replan_rules",
            "failure_codes",
            "score",
        ):
            self.assertNotIn(forbidden, first)

    def test_capability_request_is_bounded_and_uses_fixture_identity(self):
        candidate = self.manifest.candidates[0]
        fixture = CAPABILITY_FIXTURES[-1]
        request = capability_execution_request(candidate, fixture)
        self.assertEqual(request.candidate_id, candidate.candidate_id)
        self.assertEqual(request.task_id, fixture.id)
        self.assertEqual(request.round_id, "capability_4k")
        self.assertEqual((request.context_limit, request.output_limit), (4096, 512))
        self.assertEqual((request.temperature, request.cpu_threads, request.keep_alive), (0, 4, 0))

    def test_fake_campaign_executes_exact_36_candidate_major_items(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, _, provider, _ = self.engine(directory)
            result = engine.run()
        expected = [
            (candidate_id, fixture.id)
            for candidate_id in ROUND2_CANDIDATES
            for fixture in CAPABILITY_FIXTURES
        ]
        self.assertEqual(provider.calls, expected)
        self.assertEqual(len(provider.calls), 36)
        self.assertEqual(result.terminal, CapabilityTerminalState.COMPLETED_ELIGIBLE)
        self.assertEqual(len(result.ranking), 4)
        self.assertTrue(all(item.status == "ELIGIBLE" for item in result.summaries))

    def test_delay_does_not_consume_admitted_attempt(self):
        resources = FakeResources((ResourceDecision.DELAY, ResourceDecision.ALLOW))
        with tempfile.TemporaryDirectory() as directory:
            engine, store, provider, _ = self.engine(directory, resources=resources)
            fixture = CAPABILITY_FIXTURES[0]
            work = CapabilityWorkId("cap-runtime", ROUND2_CANDIDATES[0], fixture.id, fixture.version)
            record = engine.run_work(work)
        self.assertEqual(resources.calls, 2)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(record.technical_attempt_count, 1)
        self.assertEqual(record.terminal_status, "EVALUATED")

    def test_resource_pressure_retry_is_bounded_before_response(self):
        provider = FakeProvider(
            responses={CAPABILITY_FIXTURES[0].id: None},
            outcomes=(BenchmarkOutcome.RESOURCE_PRESSURE, BenchmarkOutcome.RESOURCE_PRESSURE),
        )
        with tempfile.TemporaryDirectory() as directory:
            engine, store, _, _ = self.engine(directory, provider=provider)
            fixture = CAPABILITY_FIXTURES[0]
            work = CapabilityWorkId("cap-runtime", ROUND2_CANDIDATES[0], fixture.id, fixture.version)
            record = engine.run_work(work)
        self.assertEqual(len(provider.calls), CAPABILITY_MAX_TECHNICAL_ATTEMPTS)
        self.assertEqual(record.technical_attempt_count, CAPABILITY_MAX_TECHNICAL_ATTEMPTS)
        self.assertEqual(record.terminal_status, "TECHNICAL_INCOMPLETE")
        self.assertIsNone(record.raw_response_digest)

    def test_wrong_model_answer_is_evaluated_once_and_never_retried(self):
        fixture = CAPABILITY_FIXTURES[0]
        provider = FakeProvider(responses={fixture.id: "{}"})
        with tempfile.TemporaryDirectory() as directory:
            engine, store, _, _ = self.engine(directory, provider=provider)
            work = CapabilityWorkId("cap-runtime", ROUND2_CANDIDATES[0], fixture.id, fixture.version)
            record = engine.run_work(work)
            result = store.evaluator_result(work)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(record.technical_attempt_count, 1)
        self.assertEqual(record.terminal_status, "EVALUATED")
        self.assertEqual(result.verdict.value, "CAPABILITY_INVALID")

    def test_cleanup_failure_after_generation_preserves_raw_and_forbids_evaluation(self):
        fixture = CAPABILITY_FIXTURES[0]

        class CleanupFailProvider(FakeProvider):
            def execute_prompt(self, candidate, request, resource_request):
                result = super().execute_prompt(candidate, request, resource_request)
                return SurvivalExecutionResult(
                    result.benchmark,
                    result.metrics,
                    False,
                    True,
                    True,
                    result.owned_ollama_pid,
                    result.owned_ollama_started_epoch,
                )

        provider = CleanupFailProvider()
        with tempfile.TemporaryDirectory() as directory:
            engine, store, _, _ = self.engine(directory, provider=provider)
            work = CapabilityWorkId("cap-runtime", ROUND2_CANDIDATES[0], fixture.id, fixture.version)
            record = engine.run_work(work)
            with self.assertRaises(Exception):
                store.evaluator_result(work)
        self.assertIsNotNone(record.raw_response_digest)
        self.assertEqual(record.terminal_status, "TECHNICAL_INCOMPLETE")
        self.assertIsNone(record.evaluator_result_digest)
        self.assertEqual(len(provider.calls), 1)

    def test_persisted_raw_with_cleanup_evidence_replays_evaluator_without_model(self):
        fixture = CAPABILITY_FIXTURES[0]
        with tempfile.TemporaryDirectory() as directory:
            store = CapabilityWorkStore(Path(directory))
            work = CapabilityWorkId("cap-runtime", ROUND2_CANDIDATES[0], fixture.id, fixture.version)
            store.create(work, fixture.digest)
            active = store.begin_attempt(work)
            raw = RawCapabilityResponse.create(
                work,
                fixture.digest,
                active.active_attempt_id,
                encoded(fixture.id),
            )
            store.publish_raw(raw)
            evidence_ref = store.publish_technical_evidence(
                work,
                active.active_attempt_id,
                {"cleanup_verified": True},
            )
            store.finish_attempt(work, evidence_ref)
            engine = CapabilityRuntimeEngine(
                "cap-runtime",
                store,
                self.manifest,
                NoInferenceProvider(),
                FakeResources(),
                monotonic=lambda: 0.0,
                sleep=lambda _seconds: None,
            )
            record = engine.run_work(work)
            result = store.evaluator_result(work)
        self.assertEqual(record.terminal_status, "EVALUATED")
        self.assertEqual(result.verdict.value, "CAPABILITY_VALID")


if __name__ == "__main__":
    unittest.main()