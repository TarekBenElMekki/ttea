"""Run harmless benchmark-supervisor safety checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

from ttea.resources import GpuResourceBudget, ResourceAssessment, ResourceBudget, ResourceDecision, ResourcePolicy, ResourceRequest

from .models import BenchmarkOutcome, BenchmarkSafetyPolicy
from .ollama import passive_ollama_check
from .safety import BenchmarkStateStore, CleanupFailure, IdempotentCleanup
from .supervisor import BenchmarkSupervisor, NoOpBenchmarkWorker, SleepingBenchmarkWorker


class _FakeResourceManager:
    def __init__(self, budgets: tuple[ResourceBudget, ...], resource_policy=ResourcePolicy(1, 1, 1, 80)) -> None:
        self.policy = resource_policy
        self.budgets = budgets
        self.calls = 0

    def assess(self, request: ResourceRequest) -> ResourceAssessment:
        now = datetime.now(timezone.utc)
        return ResourceAssessment(request.id, ResourceDecision.ALLOW, ("safe self-test preflight",), now,
                                  request.estimated_ram_bytes, 10, 1, 9 - request.estimated_ram_bytes,
                                  request.estimated_disk_bytes, 10, 1, 9 - request.estimated_disk_bytes,
                                  0 if request.estimated_gpu_vram_bytes else None, 10 if request.estimated_gpu_vram_bytes else None,
                                  10 if request.estimated_gpu_vram_bytes else None, 1 if request.estimated_gpu_vram_bytes else None,
                                  9 - request.estimated_gpu_vram_bytes if request.estimated_gpu_vram_bytes else None,
                                  40 if request.estimated_gpu_vram_bytes else None)

    def current_budget(self) -> ResourceBudget:
        value = self.budgets[min(self.calls, len(self.budgets) - 1)]
        self.calls += 1
        return value


def _budget(*, ram: int = 10, gpu_temperature: float | None = None, gpu_free: int = 10,
            gpu_max_temperature: float = 80, include_gpu: bool = False) -> ResourceBudget:
    gpu = (GpuResourceBudget(0, "self-test GPU", 10, gpu_free, 1, max(0, gpu_free - 1), gpu_temperature, gpu_max_temperature),) if include_gpu else ()
    return ResourceBudget(datetime.now(timezone.utc), ram, 1, max(0, ram - 1), 10, 1, 9, gpu)


def _run_case(path: Path, manager: _FakeResourceManager, worker, request: ResourceRequest, policy: BenchmarkSafetyPolicy, cleanup=None):
    return BenchmarkSupervisor(manager, policy, cleanup, path).run(worker, request)


def self_test() -> bool:
    policy = BenchmarkSafetyPolicy(0.02, 2, 1.0, 2.0, 0.2, 60, 70, 128, 16, 1)
    cpu_request = ResourceRequest("self-test", 1, 1)
    gpu_request = ResourceRequest("self-test-gpu", 1, 1, 1, 0)
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="ttea-benchmark-self-test-") as directory:
        root = Path(directory)
        result = _run_case(root / "normal.json", _FakeResourceManager((_budget(),)), NoOpBenchmarkWorker(), cpu_request, policy)
        checks.append(("harmless worker completes", result.outcome is BenchmarkOutcome.COMPLETED))
        result = _run_case(root / "timeout.json", _FakeResourceManager((_budget(),)), SleepingBenchmarkWorker(2), cpu_request, policy)
        checks.append(("sleeping worker times out", result.outcome is BenchmarkOutcome.TIMEOUT))
        result = _run_case(root / "ram.json", _FakeResourceManager((_budget(ram=0),)), SleepingBenchmarkWorker(2), cpu_request, policy)
        checks.append(("low RAM aborts", result.outcome is BenchmarkOutcome.RESOURCE_PRESSURE))
        result = _run_case(root / "thermal.json", _FakeResourceManager((_budget(include_gpu=True, gpu_temperature=70),)), SleepingBenchmarkWorker(2), gpu_request, policy)
        checks.append(("GPU thermal limit aborts", result.outcome is BenchmarkOutcome.THERMAL_LIMIT))
        result = _run_case(root / "telemetry.json", _FakeResourceManager((_budget(),)), SleepingBenchmarkWorker(2), gpu_request, policy)
        checks.append(("GPU telemetry loss aborts", result.outcome is BenchmarkOutcome.TELEMETRY_UNAVAILABLE))
        resident_request = ResourceRequest("self-test-resident-gpu", 1, 1, 8, 0)
        result = _run_case(root / "resident.json", _FakeResourceManager((_budget(include_gpu=True, gpu_temperature=40, gpu_free=1),)), SleepingBenchmarkWorker(2), resident_request, policy)
        checks.append(("resident GPU allocation does not double-count", result.outcome is BenchmarkOutcome.TIMEOUT))
        result = _run_case(root / "resource-thermal.json", _FakeResourceManager((_budget(include_gpu=True, gpu_temperature=65, gpu_max_temperature=65),), resource_policy=None), SleepingBenchmarkWorker(2), gpu_request, policy)
        checks.append(("effective resource thermal limit aborts", result.outcome is BenchmarkOutcome.THERMAL_LIMIT))
        calls = []
        cleanup = IdempotentCleanup((lambda: calls.append("cleaned"),))
        _run_case(root / "cleanup.json", _FakeResourceManager((_budget(),)), NoOpBenchmarkWorker(), cpu_request, policy, cleanup)
        cleanup.run()
        checks.append(("cleanup is logically once", calls == ["cleaned"]))
        fallback = []
        bounded = IdempotentCleanup((lambda: time.sleep(0.2), lambda: fallback.append("ran")))
        try:
            bounded.run(0.02)
        except CleanupFailure:
            cleanup_bounded = fallback == ["ran"]
        else:
            cleanup_bounded = False
        checks.append(("cleanup timeout preserves fallback", cleanup_bounded))
        state_path = root / "ownership.json"
        owner = BenchmarkStateStore(state_path)
        owner.acquire()
        try:
            try:
                BenchmarkStateStore(state_path).acquire()
            except RuntimeError:
                identity_blocked = True
            else:
                identity_blocked = False
        finally:
            owner.release()
        checks.append(("active supervisor identity blocks acquisition", identity_blocked))
        state_path.write_text(json.dumps({"supervisor_pid": os.getpid(), "supervisor_started_epoch": 0, "worker_pid": None}), encoding="utf-8")
        stale = BenchmarkStateStore(state_path)
        stale.acquire()
        stale.release()
        checks.append(("mismatched stale identity is reclaimed", not state_path.exists()))
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    return all(passed for _, passed in checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="T/TEA benchmark safety supervisor")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true", help="run harmless supervision checks")
    modes.add_argument("--ollama-check", action="store_true", help="passively check local Ollama availability")
    arguments = parser.parse_args()
    if arguments.ollama_check:
        executable, version, api, models = passive_ollama_check()
        print("T/TEA Ollama Check\n------------------")
        print(f"Executable: {executable}\nVersion: {version}\nDefault API: {api}")
        installed = "unavailable" if models is None else (", ".join(models) or "(none)")
        print(f"Models installed: {installed}")
    elif not self_test():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
