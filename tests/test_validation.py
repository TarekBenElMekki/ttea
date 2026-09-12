from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ttea.benchmarking import BenchmarkSafetyPolicy, BenchmarkStateStore
from ttea.benchmarking.ollama import OwnedProcessIdentity
from ttea.validation import (
    ScenarioCheck, ScenarioInterrupted, ScenarioVerification, SuiteRegistry,
    ValidationRunResult, ValidationRunner, ValidationScenario, ValidationScenarioResult,
    ValidationStatus, ValidationSuite, exit_code, freeze_evidence, load_latest_report,
    text_report,
)
from ttea.validation.reporting import result_to_dict
from ttea.validation.suites import (
    DedicatedPortPrecheckScenario, DefaultServerIsolationScenario, PreModelContext,
    stop_verified_process,
)


class PassScenario(ValidationScenario):
    scenario_id = "pass"

    def execute(self, context):
        context.append("execute")
        return {"metric": 1}


class FailScenario(ValidationScenario):
    scenario_id = "fail"

    def verify(self, context, execution):
        return ScenarioVerification(ValidationStatus.FAIL, ("invariant failed",))


class BlockedScenario(ValidationScenario):
    scenario_id = "blocked"

    def precheck(self, context):
        return ScenarioCheck(ValidationStatus.BLOCKED, ("resource pressure",))


class SkipScenario(ValidationScenario):
    scenario_id = "skip"

    def precheck(self, context):
        return ScenarioCheck(ValidationStatus.SKIP, ("not applicable",))


class ExceptionScenario(ValidationScenario):
    scenario_id = "exception"

    def execute(self, context):
        raise RuntimeError("boom")

    def cleanup(self, context):
        context.append("cleanup")


class InterruptScenario(ExceptionScenario):
    scenario_id = "interrupt"

    def execute(self, context):
        raise KeyboardInterrupt()


class RetryScenario(ValidationScenario):
    scenario_id = "retry"

    def precheck(self, context):
        context["calls"] += 1
        if context["calls"] == 1:
            return ScenarioCheck(ValidationStatus.BLOCKED, ("temporary",), retryable_resource_delay=True)
        return ScenarioCheck()


class ValidationFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def result(self, status=ValidationStatus.PASS):
        now = datetime.now(timezone.utc)
        scenario = ValidationScenarioResult("one", status, ("reason",), now, now, freeze_evidence({"value": 1}))
        return ValidationRunResult("run", "suite", now, now, status, (scenario,), freeze_evidence({"host": "test"}), True)

    def test_pass_fail_blocked_skip_and_exception_conversion(self):
        context = []
        self.assertIs(PassScenario().run(context).status, ValidationStatus.PASS)
        self.assertIs(FailScenario().run(context).status, ValidationStatus.FAIL)
        self.assertIs(BlockedScenario().run(context).status, ValidationStatus.BLOCKED)
        self.assertIs(SkipScenario().run(context).status, ValidationStatus.SKIP)
        converted = ExceptionScenario().run(context)
        self.assertIs(converted.status, ValidationStatus.FAIL)
        self.assertIn("RuntimeError", converted.reasons[0])
        self.assertIn("cleanup", context)

    def test_cleanup_on_keyboard_interrupt(self):
        context = []
        with self.assertRaises(ScenarioInterrupted) as captured:
            InterruptScenario().run(context)
        self.assertEqual(context, ["cleanup"])
        self.assertTrue(captured.exception.result.reasons[0].startswith("validation interrupted"))

    def test_report_json_text_latest_and_partial_failure(self):
        registry = SuiteRegistry()
        registry.register(ValidationSuite("mixed", (PassScenario, ExceptionScenario), list))
        result = ValidationRunner(registry, self.root).run("mixed")
        self.assertIs(result.overall_status, ValidationStatus.FAIL)
        self.assertEqual(len(result.scenario_results), 2)
        payload = json.loads(Path(result.report_paths[0]).read_text(encoding="utf-8"))
        self.assertEqual(payload["overall_status"], "FAIL")
        self.assertEqual(len(payload["scenario_results"]), 2)
        self.assertIn("T/TEA Validation", Path(result.report_paths[1]).read_text(encoding="utf-8"))
        self.assertEqual(load_latest_report(self.root)["run_id"], result.run_id)
        self.assertIn("Cleanup verified: YES", text_report(result))

    def test_immutable_evidence_preserves_empty_json_types(self):
        now = datetime.now(timezone.utc)
        scenario = ValidationScenarioResult("typed", ValidationStatus.PASS, ("ok",), now, now,
                                            freeze_evidence({"array": [], "object": {}}))
        run = ValidationRunResult("typed", "suite", now, now, ValidationStatus.PASS, (scenario,),
                                  freeze_evidence({}), True)
        evidence = result_to_dict(run)["scenario_results"][0]["evidence"]
        self.assertEqual(evidence["array"], [])
        self.assertEqual(evidence["object"], {})

    def test_stable_exit_codes(self):
        self.assertEqual(exit_code(self.result(ValidationStatus.PASS)), 0)
        self.assertEqual(exit_code(self.result(ValidationStatus.FAIL)), 1)
        self.assertEqual(exit_code(self.result(ValidationStatus.BLOCKED)), 2)
        interrupted = self.result(ValidationStatus.FAIL)
        interrupted = ValidationRunResult(interrupted.run_id, interrupted.suite, interrupted.started_at,
                                          interrupted.finished_at, interrupted.overall_status,
                                          interrupted.scenario_results, interrupted.environment_summary, True,
                                          interrupted=True)
        self.assertEqual(exit_code(interrupted), 130)

    def test_suite_registration(self):
        registry = SuiteRegistry()
        suite = ValidationSuite("sample", (PassScenario,), list)
        registry.register(suite)
        self.assertEqual(registry.names(), ("sample",))
        self.assertIs(registry.get("sample"), suite)
        with self.assertRaises(ValueError):
            registry.register(suite)

    def test_resource_wait_retries_but_default_never_waits(self):
        context = {"calls": 0}
        result = RetryScenario().run(context, 0.02, retry_interval_seconds=0.001)
        self.assertIs(result.status, ValidationStatus.PASS)
        self.assertEqual(context["calls"], 2)
        context = {"calls": 0}
        result = RetryScenario().run(context, 0)
        self.assertIs(result.status, ValidationStatus.BLOCKED)
        self.assertEqual(context["calls"], 1)

    def test_occupied_dedicated_port_blocks_without_termination(self):
        context = Mock(dedicated_port_was_open=True)
        with patch("ttea.validation.suites.listening_identity", return_value=None):
            result = DedicatedPortPrecheckScenario().run(context)
        self.assertIs(result.status, ValidationStatus.BLOCKED)

    def test_exact_owned_stop_never_touches_unrelated_process(self):
        class FakeProcess:
            def __init__(self, pid, created):
                self.pid, self.created, self.terminated = pid, created, False
            def create_time(self): return self.created
            def terminate(self): self.terminated = True
            def wait(self, timeout): return 0
            def kill(self): self.terminated = True
        owned, unrelated = FakeProcess(10, 20), FakeProcess(11, 21)
        stop_verified_process(OwnedProcessIdentity(10, 20), process_api=lambda pid: owned)
        self.assertTrue(owned.terminated)
        self.assertFalse(unrelated.terminated)

    def test_default_ollama_identity_preserved(self):
        identity = OwnedProcessIdentity(os.getpid(), 10.0)
        context = Mock(default_port_was_open=True, default_identity=identity)
        with patch("ttea.validation.suites.listening_identity", return_value=identity):
            result = DefaultServerIsolationScenario().run(context)
        self.assertIs(result.status, ValidationStatus.PASS)

    def test_global_guard_failure_still_writes_report(self):
        class BadGuard:
            def establish(self): raise RuntimeError("active owned process")
        registry = SuiteRegistry()
        registry.register(ValidationSuite("guarded", (PassScenario,), list, lambda context: BadGuard()))
        result = ValidationRunner(registry, self.root).run("guarded")
        self.assertIs(result.overall_status, ValidationStatus.BLOCKED)
        self.assertTrue(Path(result.report_paths[0]).is_file())


if __name__ == "__main__":
    unittest.main()
