from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

from ttea.benchmarking import BenchmarkOutcome, BenchmarkSafetyPolicy, BenchmarkStateStore, BenchmarkSupervisor, CleanupFailure, FailingBenchmarkWorker, IdempotentCleanup, NoOpBenchmarkWorker, SleepingBenchmarkWorker
from ttea.resources import GpuResourceBudget, ResourceAssessment, ResourceBudget, ResourceDecision, ResourcePolicy, ResourceRequest


def budget(ram=10, gpu=None):
    return ResourceBudget(datetime.now(timezone.utc), ram, 1, max(0, ram - 1), 10, 1, 9, () if gpu is None else (gpu,))


def gpu(free=10, temperature=40, max_temperature=80):
    return GpuResourceBudget(0, "gpu", 10, free, 1, max(0, free - 1), temperature, max_temperature)


class FakeManager:
    def __init__(self, budgets, decision=ResourceDecision.ALLOW, resource_policy=ResourcePolicy(1, 1, 1, 80)):
        self.policy = resource_policy
        self.budgets, self.decision, self.calls = list(budgets), decision, 0

    def assess(self, request):
        now = datetime.now(timezone.utc)
        return ResourceAssessment(request.id, self.decision, ("preflight",), now, request.estimated_ram_bytes, 10, 1, 9, request.estimated_disk_bytes, 10, 1, 9, 0 if request.estimated_gpu_vram_bytes else None)

    def current_budget(self):
        result = self.budgets[min(self.calls, len(self.budgets) - 1)]
        self.calls += 1
        return result


class ResultWorker(NoOpBenchmarkWorker):
    def run(self):
        return ("structured", 7)


class BenchmarkingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "state.json"
        self.policy = BenchmarkSafetyPolicy(0.02, 2, 1.0, 2.0, 0.2, 60, 70, 128, 16, 1)
        self.request = ResourceRequest("test", 1, 1)

    def supervisor(self, manager, cleanup=None):
        return BenchmarkSupervisor(manager, self.policy, cleanup, self.path)

    def test_policy_validation_and_resource_policy_compatibility(self):
        for values in ({"telemetry_interval_seconds": 0}, {"consecutive_resource_violation_samples": 0}, {"max_context_tokens": 4097}, {"gpu_warning_temperature_c": 70, "gpu_abort_temperature_c": 70}, {"model_timeout_seconds": -1}):
            with self.assertRaises(ValueError): BenchmarkSafetyPolicy(**values)
        with self.assertRaises(ValueError): self.policy.validate_with(ResourcePolicy(1, 1, 1, 69))

    def test_normal_completion_timeout_and_worker_failure(self):
        self.assertEqual(self.supervisor(FakeManager([budget()])).run(NoOpBenchmarkWorker(), self.request).outcome, BenchmarkOutcome.COMPLETED)
        self.assertEqual(self.supervisor(FakeManager([budget()])).run(SleepingBenchmarkWorker(2), self.request).outcome, BenchmarkOutcome.TIMEOUT)
        self.assertEqual(self.supervisor(FakeManager([budget()])).run(FailingBenchmarkWorker(), self.request).outcome, BenchmarkOutcome.WORKER_FAILURE)
        self.assertFalse(self.path.exists())

    def test_worker_result_crosses_supervisor_boundary(self):
        result = self.supervisor(FakeManager([budget()])).run(ResultWorker(), self.request)
        self.assertEqual(result.outcome, BenchmarkOutcome.COMPLETED)
        self.assertEqual(result.worker_result, ("structured", 7))

    def test_preflight_must_allow(self):
        result = self.supervisor(FakeManager([budget()], ResourceDecision.DELAY)).run(NoOpBenchmarkWorker(), self.request)
        self.assertEqual(result.outcome, BenchmarkOutcome.RESOURCE_PRESSURE)
        self.assertIsNone(result.worker_pid)

    def test_consecutive_ram_and_runtime_reserve_pressure(self):
        result = self.supervisor(FakeManager([budget(0)])).run(SleepingBenchmarkWorker(2), self.request)
        self.assertEqual(result.outcome, BenchmarkOutcome.RESOURCE_PRESSURE)
        gpu_request = ResourceRequest("gpu", 1, 1, 8, 0)
        resident = self.supervisor(FakeManager([budget(gpu=gpu(free=1))])).run(SleepingBenchmarkWorker(2), gpu_request)
        self.assertEqual(resident.outcome, BenchmarkOutcome.TIMEOUT)
        result = self.supervisor(FakeManager([budget(gpu=gpu(free=0))])).run(SleepingBenchmarkWorker(2), gpu_request)
        self.assertEqual(result.outcome, BenchmarkOutcome.RESOURCE_PRESSURE)

    def test_thermal_and_gpu_telemetry_abort_immediately(self):
        gpu_request = ResourceRequest("gpu", 1, 1, 1, 0)
        result = self.supervisor(FakeManager([budget(gpu=gpu(temperature=70))])).run(SleepingBenchmarkWorker(2), gpu_request)
        self.assertEqual(result.outcome, BenchmarkOutcome.THERMAL_LIMIT)
        result = self.supervisor(FakeManager([budget()])).run(SleepingBenchmarkWorker(2), gpu_request)
        self.assertEqual(result.outcome, BenchmarkOutcome.TELEMETRY_UNAVAILABLE)

    def test_stricter_effective_thermal_boundary_aborts_immediately(self):
        gpu_request = ResourceRequest("gpu", 1, 1, 1, 0)
        resource_stricter = FakeManager([budget(gpu=gpu(temperature=65, max_temperature=65))], resource_policy=None)
        result = self.supervisor(resource_stricter).run(SleepingBenchmarkWorker(2), gpu_request)
        self.assertEqual(result.outcome, BenchmarkOutcome.THERMAL_LIMIT)
        benchmark_stricter = FakeManager([budget(gpu=gpu(temperature=70, max_temperature=75))], resource_policy=None)
        result = self.supervisor(benchmark_stricter).run(SleepingBenchmarkWorker(2), gpu_request)
        self.assertEqual(result.outcome, BenchmarkOutcome.THERMAL_LIMIT)

    def test_cleanup_once_for_completion_timeout_failure_and_interrupt(self):
        for worker in (NoOpBenchmarkWorker(), SleepingBenchmarkWorker(2), FailingBenchmarkWorker()):
            calls = []
            cleanup = IdempotentCleanup((lambda: calls.append(1),))
            self.supervisor(FakeManager([budget()]), cleanup).run(worker, self.request)
            cleanup.run()
            self.assertEqual(calls, [1])
        calls = []
        cleanup = IdempotentCleanup((lambda: calls.append(1),))
        supervisor = self.supervisor(FakeManager([budget()]), cleanup)
        with patch.object(supervisor, "_runtime_violation", side_effect=KeyboardInterrupt):
            result = supervisor.run(SleepingBenchmarkWorker(2), self.request)
        self.assertEqual(result.outcome, BenchmarkOutcome.CANCELLED)
        self.assertEqual(calls, [1])

    def test_cleanup_runs_for_each_safety_abort(self):
        gpu_request = ResourceRequest("gpu", 1, 1, 1, 0)
        cases = (
            (FakeManager([budget(0)]), self.request, BenchmarkOutcome.RESOURCE_PRESSURE),
            (FakeManager([budget(gpu=gpu(temperature=70))]), gpu_request, BenchmarkOutcome.THERMAL_LIMIT),
            (FakeManager([budget()]), gpu_request, BenchmarkOutcome.TELEMETRY_UNAVAILABLE),
        )
        for manager, request, expected in cases:
            calls = []
            result = self.supervisor(manager, IdempotentCleanup((lambda: calls.append(1),))).run(SleepingBenchmarkWorker(2), request)
            self.assertEqual(result.outcome, expected)
            self.assertEqual(calls, [1])

    def test_stale_or_invalid_state_never_targets_unrelated_process(self):
        self.path.write_text(json.dumps({"supervisor_pid": 999, "worker_pid": os.getpid(), "worker_started_epoch": 0}), encoding="utf-8")
        store = BenchmarkStateStore(self.path)
        self.assertFalse(store.process_matches(os.getpid(), 0))
        store.acquire()
        self.assertTrue(self.path.exists())
        store.release()
        self.assertFalse(self.path.exists())

    def test_active_supervisor_without_worker_blocks_acquisition(self):
        owner = BenchmarkStateStore(self.path)
        owner.acquire()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(state["worker_pid"])
        with self.assertRaises(RuntimeError): BenchmarkStateStore(self.path).acquire()
        owner.release()

    def test_active_recorded_worker_blocks_when_supervisor_is_stale(self):
        started = psutil.Process(os.getpid()).create_time()
        self.path.write_text(json.dumps({"supervisor_pid": 999999, "supervisor_started_epoch": 1,
                                         "worker_pid": os.getpid(), "worker_started_epoch": started}), encoding="utf-8")
        with self.assertRaises(RuntimeError): BenchmarkStateStore(self.path).acquire()

    def test_fully_stale_state_is_reclaimed_and_creation_mismatch_is_stale(self):
        self.path.write_text(json.dumps({"supervisor_pid": 999999, "supervisor_started_epoch": 1,
                                         "worker_pid": 999998, "worker_started_epoch": 1}), encoding="utf-8")
        store = BenchmarkStateStore(self.path)
        store.acquire()
        store.release()
        mismatch = psutil.Process(os.getpid()).create_time() + 1000
        self.path.write_text(json.dumps({"supervisor_pid": os.getpid(), "supervisor_started_epoch": mismatch,
                                         "worker_pid": None}), encoding="utf-8")
        store.acquire()
        store.release()

    def test_surviving_owned_worker_preserves_state(self):
        class FakeQueue:
            def get_nowait(self): raise queue.Empty
            def get(self, timeout=None): raise queue.Empty
            def close(self): pass
            def cancel_join_thread(self): pass
        class SurvivingProcess:
            pid = os.getpid()
            exitcode = None
            def start(self): pass
            def is_alive(self): return True
            def terminate(self): pass
            def kill(self): pass
            def join(self, timeout=None): pass
        class FakeContext:
            def Queue(self): return FakeQueue()
            def Process(self, **kwargs): return SurvivingProcess()
        short_policy = BenchmarkSafetyPolicy(0.005, 2, 0.01, 0.02, 0.01, 60, 70, 128, 16, 1)
        supervisor = BenchmarkSupervisor(FakeManager([budget()]), short_policy, state_path=self.path, multiprocessing_context=FakeContext())
        result = supervisor.run(NoOpBenchmarkWorker(), self.request)
        self.assertEqual(result.outcome, BenchmarkOutcome.WORKER_FAILURE)
        self.assertTrue(self.path.exists())
        self.assertIn("termination could not be confirmed", " ".join(result.reasons))
        blocked = BenchmarkSupervisor(FakeManager([budget()]), short_policy, state_path=self.path, multiprocessing_context=FakeContext()).run(NoOpBenchmarkWorker(), self.request)
        self.assertEqual(blocked.outcome, BenchmarkOutcome.WORKER_FAILURE)
        self.assertIsNone(blocked.worker_pid)

    def test_cleanup_steps_are_bounded_continue_and_remain_idempotent(self):
        calls = []
        cleanup = IdempotentCleanup((lambda: time.sleep(0.2), lambda: calls.append("fallback")))
        started = time.monotonic()
        with self.assertRaises(CleanupFailure) as captured: cleanup.run(0.02)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.15)
        self.assertEqual(calls, ["fallback"])
        self.assertIn("timed out", str(captured.exception))
        cleanup.run(0.02)
        self.assertEqual(calls, ["fallback"])

    def test_cleanup_failure_changes_completed_result_to_non_success(self):
        def fail():
            raise RuntimeError("cleanup broke")
        result = self.supervisor(FakeManager([budget()]), IdempotentCleanup((fail,))).run(NoOpBenchmarkWorker(), self.request)
        self.assertEqual(result.outcome, BenchmarkOutcome.WORKER_FAILURE)
        self.assertIn("cleanup failed", " ".join(result.reasons))


if __name__ == "__main__":
    unittest.main()
