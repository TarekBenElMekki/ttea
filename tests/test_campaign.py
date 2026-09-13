from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from ttea.resources import ResourceDecision
from ttea.validation import (
    CampaignManager, CampaignState, CampaignStore, CampaignSupervisor, SuiteRegistry,
    ValidationCampaign, ValidationRunResult, ValidationScenarioResult, ValidationStatus,
    ValidationSuite, campaign_from_dict, campaign_to_dict, freeze_evidence,
)
from ttea.validation.campaign import requested_suites, retryable_resource_delay


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def campaign(state=CampaignState.CREATED, *, campaign_id="campaign-test", pid=100, started=10.0,
             deadline=None, history=(), cancel_requested=False):
    return ValidationCampaign(
        campaign_id, ("pre_model",), "pre_model", "pre_model", state, NOW, NOW,
        deadline or NOW + timedelta(hours=1), pid, started, NOW, retry_count=0,
        poll_interval_seconds=30.0, cancel_requested=cancel_requested, history=history,
    )


def validation_result(status, decision=None, *, interrupted=False, cleanup=True, run_id="run-1"):
    now = NOW + timedelta(minutes=1)
    evidence = {} if decision is None else {"resource_decision": decision.value}
    scenario = ValidationScenarioResult("scenario", status, (f"{status.value.lower()} reason",), NOW, now,
                                        freeze_evidence(evidence))
    return ValidationRunResult(run_id, "pre_model", NOW, now, status, (scenario,), freeze_evidence({}),
                               cleanup, interrupted=interrupted)


class FakeRunner:
    def __init__(self, values):
        self.values, self.calls, self.options = list(values), [], []
    def run(self, suite, **kwargs):
        self.calls.append(suite)
        self.options.append(kwargs.get("options"))
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeProcess:
    def __init__(self, pid, created):
        self.pid, self.created = pid, created
        self.terminated = self.killed = False
    def create_time(self): return self.created
    def terminate(self): self.terminated = True
    def kill(self): self.killed = True
    def wait(self, timeout): return 0


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def store(self, matches=lambda pid, started: False):
        return CampaignStore(self.root / "campaigns", matches)

    def test_campaign_state_validation_and_round_trip(self):
        value = campaign()
        self.assertEqual(campaign_from_dict(campaign_to_dict(value)), value)
        with self.assertRaises(ValueError):
            replace(value, requested_suites=())
        with self.assertRaises(ValueError):
            replace(value, supervisor_pid=True)
        with self.assertRaises(ValueError):
            replace(value, poll_interval_seconds=0)

    def test_atomic_persistence_and_history(self):
        store = self.store()
        value = campaign(pid=None, started=None)
        store.create(value)
        loaded = store.load(value.campaign_id)
        self.assertEqual(loaded, value)
        self.assertFalse(tuple(store.campaign_dir(value.campaign_id).glob("*.tmp")))
        self.assertEqual(json.loads(store.state_path(value.campaign_id).read_text(encoding="utf-8"))["state"], "CREATED")

    def test_only_one_active_campaign_and_stale_reclaim(self):
        live_store = self.store(lambda pid, started: pid == 100 and started == 10.0)
        live_store.create(campaign())
        with self.assertRaises(RuntimeError):
            live_store.create(campaign(campaign_id="campaign-second"))
        stale_store = CampaignStore(self.root / "stale", lambda pid, started: False)
        stale_store.create(campaign(campaign_id="old"))
        stale_store.create(campaign(campaign_id="new"))
        self.assertIs(stale_store.load("old").state, CampaignState.INTERRUPTED)
        self.assertEqual(stale_store.active_id(), "new")

    def test_requested_suite_progression_requires_registered_predecessors(self):
        registry = SuiteRegistry()
        registry.register(ValidationSuite("pre_model", (), list))
        self.assertEqual(requested_suites(registry, "pre_model"), ("pre_model",))
        with self.assertRaises(ValueError):
            requested_suites(registry, "tiny_model")

    def test_detached_spawn_arguments_are_safe_and_record_identity(self):
        store = self.store(lambda pid, started: pid == 222 and started == 22.0)
        spawned = FakeProcess(222, 22.0)
        def spawn(command, **kwargs):
            current = store.current()
            store.save(replace(current, supervisor_pid=222, supervisor_started_epoch=22.0,
                               supervisor_claim_token=None))
            return spawned
        process_factory = Mock(side_effect=spawn)
        process_api = lambda pid: FakeProcess(pid, 11.0 if pid == os.getpid() else 22.0)
        registry = SuiteRegistry()
        registry.register(ValidationSuite("pre_model", (), list))
        manager = CampaignManager(store, registry, self.root / "reports", process_factory, process_api,
                                  benchmark_state_path=self.root / "benchmark.json")
        result = manager.start("pre_model", detach=True, allow_model_download=True)
        self.assertEqual(result.supervisor_pid, 222)
        self.assertTrue(result.allow_model_download)
        command = process_factory.call_args.args[0]
        self.assertEqual(command[:3], [os.sys.executable, "-m", "ttea.validation"])
        self.assertIn("_supervise", command)
        self.assertIn("--claim-token", command)
        self.assertIs(process_factory.call_args.kwargs["stdin"], __import__("subprocess").DEVNULL)
        self.assertIs(process_factory.call_args.kwargs["stderr"], __import__("subprocess").STDOUT)
        self.assertNotIn("shell", process_factory.call_args.kwargs)

    def test_waiting_delay_retry_deny_pass_fail_and_history(self):
        store = self.store()
        base = store.create(campaign(pid=None, started=None))
        supervisor = CampaignSupervisor(base.campaign_id, store, FakeRunner([]), now=lambda: NOW + timedelta(minutes=2))
        waiting = supervisor.record_result(base, validation_result(ValidationStatus.BLOCKED, ResourceDecision.DELAY))
        self.assertIs(waiting.state, CampaignState.WAITING_FOR_RESOURCES)
        self.assertEqual(waiting.retry_count, 1)
        self.assertEqual(waiting.history[0].run_id, "run-1")
        self.assertIsNone(waiting.wait_periods[0].finished_at)
        denied = supervisor.record_result(base, validation_result(ValidationStatus.BLOCKED, ResourceDecision.DENY))
        self.assertIs(denied.state, CampaignState.BLOCKED)
        completed = supervisor.record_result(base, validation_result(ValidationStatus.PASS))
        self.assertIs(completed.state, CampaignState.COMPLETED)
        failed = supervisor.record_result(base, validation_result(ValidationStatus.FAIL, cleanup=False))
        self.assertIs(failed.state, CampaignState.FAILED)
        self.assertTrue((store.campaign_dir(base.campaign_id) / "campaign-report.json").is_file())

    def test_retry_classifier_requires_exclusive_delay(self):
        self.assertTrue(retryable_resource_delay(validation_result(ValidationStatus.BLOCKED, ResourceDecision.DELAY)))
        self.assertFalse(retryable_resource_delay(validation_result(ValidationStatus.BLOCKED, ResourceDecision.DENY)))
        self.assertFalse(retryable_resource_delay(validation_result(ValidationStatus.BLOCKED)))

    def test_timeout_stops_campaign_and_heartbeat_updates(self):
        store = self.store()
        value = campaign(deadline=NOW + timedelta(seconds=1), pid=None, started=None)
        store.create(value)
        supervisor = CampaignSupervisor(value.campaign_id, store, FakeRunner([]),
                                        now=lambda: NOW + timedelta(seconds=2))
        result = supervisor.run()
        self.assertIs(result.state, CampaignState.BLOCKED)
        self.assertEqual(result.heartbeat_at, NOW + timedelta(seconds=2))
        self.assertIn("maximum duration", result.last_reason)

    def test_status_live_and_stale(self):
        live_store = self.store(lambda pid, started: True)
        live_store.create(campaign())
        live, alive = CampaignManager(live_store).status()
        self.assertTrue(alive)
        self.assertIs(live.state, CampaignState.CREATED)
        stale_store = CampaignStore(self.root / "stale-status", lambda pid, started: False)
        stale_store.create(campaign())
        stale, alive = CampaignManager(stale_store).status()
        self.assertFalse(alive)
        self.assertIs(stale.state, CampaignState.INTERRUPTED)

    def test_cancel_verifies_identity_and_never_targets_stale_pid(self):
        owned = FakeProcess(100, 10.0)
        calls = []
        matches = Mock(return_value=True)
        store = self.store(matches)
        store.create(campaign())
        manager = CampaignManager(store, process_api=lambda pid: calls.append(pid) or owned, sleep=lambda seconds: None)
        cancelled = manager.cancel(graceful_timeout_seconds=0)
        self.assertTrue(owned.terminated)
        self.assertEqual(calls, [100])
        self.assertIs(cancelled.state, CampaignState.CANCELLED)

        stale_store = CampaignStore(self.root / "cancel-stale", lambda pid, started: False)
        stale_store.create(campaign())
        unrelated_api = Mock(side_effect=AssertionError("must not inspect or terminate unrelated PID"))
        stale_manager = CampaignManager(stale_store, process_api=unrelated_api)
        self.assertIs(stale_manager.cancel().state, CampaignState.CANCELLED)
        unrelated_api.assert_not_called()

    def test_resume_stale_campaign_and_active_cannot_resume_twice(self):
        store = self.store(lambda pid, started: pid == 333 and started == 33.0)
        store.create(campaign(state=CampaignState.INTERRUPTED))
        child = FakeProcess(333, 33.0)
        process_api = lambda pid: FakeProcess(pid, 11.0 if pid == os.getpid() else 33.0)
        registry = SuiteRegistry()
        registry.register(ValidationSuite("pre_model", (), list))
        def spawn(command, **kwargs):
            current = store.current()
            store.save(replace(current, supervisor_pid=333, supervisor_started_epoch=33.0,
                               supervisor_claim_token=None))
            return child
        manager = CampaignManager(store, registry, self.root / "reports", Mock(side_effect=spawn), process_api,
                                  benchmark_state_path=self.root / "benchmark.json")
        resumed = manager.resume()
        self.assertEqual(resumed.supervisor_pid, 333)
        active_store = CampaignStore(self.root / "active", lambda pid, started: True)
        active_store.create(campaign())
        with self.assertRaises(RuntimeError):
            CampaignManager(active_store).resume()

    def test_interruption_persists_recoverable_state(self):
        store = self.store()
        value = store.create(campaign(pid=None, started=None))
        supervisor = CampaignSupervisor(value.campaign_id, store, FakeRunner([KeyboardInterrupt()]), now=lambda: NOW)
        result = supervisor.run()
        self.assertIs(result.state, CampaignState.INTERRUPTED)
        self.assertIs(store.load(value.campaign_id).state, CampaignState.INTERRUPTED)

    def test_download_authorization_persists_and_pre_model_pass_advances(self):
        store = self.store()
        value = ValidationCampaign(
            "campaign-through-tiny", ("pre_model", "tiny_model"), "tiny_model", "pre_model",
            CampaignState.CREATED, NOW, NOW, NOW + timedelta(hours=1), None, None, NOW,
            allow_model_download=True,
        )
        store.create(value)
        self.assertTrue(store.load(value.campaign_id).allow_model_download)
        runner = FakeRunner([validation_result(ValidationStatus.PASS, run_id="pre-pass"),
                             validation_result(ValidationStatus.PASS, run_id="tiny-pass")])
        completed = CampaignSupervisor(value.campaign_id, store, runner, now=lambda: NOW).run()
        self.assertIs(completed.state, CampaignState.COMPLETED)
        self.assertEqual(runner.calls, ["pre_model", "tiny_model"])
        self.assertFalse(runner.options[0].pre_model_passed)
        self.assertTrue(runner.options[1].pre_model_passed)
        self.assertTrue(runner.options[1].allow_model_download)

    def test_no_process_name_termination_implementation(self):
        source = (Path(__file__).parents[1] / "src" / "ttea" / "validation" / "campaign.py").read_text(encoding="utf-8")
        self.assertNotIn("taskkill", source.lower())
        self.assertNotIn("Stop-Process", source)
        self.assertNotIn("process_iter", source)


if __name__ == "__main__":
    unittest.main()
