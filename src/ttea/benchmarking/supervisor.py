"""Independent child-process supervision for bounded benchmark stubs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import queue
import time

from ttea.logging import get_logger, setup_logging
from ttea.resources import ResourceDecision, ResourceManager, ResourceRequest

from .models import BenchmarkOutcome, BenchmarkResult, BenchmarkSafetyPolicy
from .safety import BenchmarkStateStore, IdempotentCleanup

LOGGER = get_logger("benchmarking.supervisor")


class WorkerTerminationError(RuntimeError):
    """The supervisor could not prove that its child stopped."""


class BenchmarkWorker(ABC):
    """Pickle-safe contract for a future single-model, single-request worker."""

    @abstractmethod
    def run(self) -> None:
        pass


class NoOpBenchmarkWorker(BenchmarkWorker):
    def run(self) -> None:
        return None


class SleepingBenchmarkWorker(BenchmarkWorker):
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def run(self) -> None:
        time.sleep(self.seconds)


class FailingBenchmarkWorker(BenchmarkWorker):
    def run(self) -> None:
        raise RuntimeError("intentional stub worker failure")


def _worker_entry(worker: BenchmarkWorker, messages) -> None:
    try:
        messages.put(("started", None))
        result = worker.run()
    except BaseException as error:
        messages.put(("failure", f"{type(error).__name__}: {error}"))
        raise
    else:
        messages.put(("completed", result))


class BenchmarkSupervisor:
    def __init__(self, resource_manager: ResourceManager, policy: BenchmarkSafetyPolicy | None = None,
                 cleanup: IdempotentCleanup | None = None, state_path: Path = Path("data/runtime/benchmark.json"),
                 multiprocessing_context=None, state_store: BenchmarkStateStore | None = None,
                 state_already_acquired: bool = False, worker_started_callback=None,
                 telemetry_observer=None) -> None:
        self.resource_manager = resource_manager
        self.policy = policy or BenchmarkSafetyPolicy()
        if resource_manager.policy is not None:
            self.policy.validate_with(resource_manager.policy)
        self.cleanup = cleanup or IdempotentCleanup()
        self.state_store = state_store or BenchmarkStateStore(state_path)
        self.state_already_acquired = state_already_acquired
        self.worker_started_callback = worker_started_callback
        self.telemetry_observer = telemetry_observer
        self.context = multiprocessing_context or multiprocessing.get_context("spawn")
        self._active_process = None
        setup_logging()

    def _stop_owned_worker(self, process) -> None:
        try:
            if not process.is_alive():
                process.join(timeout=0)
                return
            process.terminate()
            process.join(timeout=self.policy.cooldown_timeout_seconds)
            if process.is_alive():
                process.kill()
                process.join(timeout=self.policy.cooldown_timeout_seconds)
            if process.is_alive():
                raise WorkerTerminationError(f"owned benchmark worker PID {process.pid} remains alive after terminate and kill")
        except WorkerTerminationError:
            raise
        except Exception as error:
            raise WorkerTerminationError(f"owned benchmark worker PID {process.pid} termination failed: {error}") from error

    def _runtime_violation(self, request: ResourceRequest, selected_gpu_index: int | None,
                           ram_violations: int, vram_violations: int):
        try:
            budget = self.resource_manager.current_budget()
        except Exception as error:
            return BenchmarkOutcome.TELEMETRY_UNAVAILABLE, (f"runtime telemetry unavailable: {error}",), ram_violations, vram_violations

        if self.telemetry_observer is not None:
            try:
                self.telemetry_observer(budget)
            except Exception as error:
                return BenchmarkOutcome.TELEMETRY_UNAVAILABLE, (f"runtime telemetry observer failed: {error}",), ram_violations, vram_violations

        ram_violations = ram_violations + 1 if budget.available_ram_bytes < budget.reserved_ram_bytes else 0
        if ram_violations >= self.policy.consecutive_resource_violation_samples:
            return BenchmarkOutcome.RESOURCE_PRESSURE, ("available RAM remained below the reserved RAM threshold",), ram_violations, vram_violations

        if request.estimated_gpu_vram_bytes:
            gpu = next((item for item in budget.gpus if item.index == selected_gpu_index), None)
            if gpu is None or gpu.free_vram_bytes is None or gpu.temperature_c is None:
                return BenchmarkOutcome.TELEMETRY_UNAVAILABLE, ("required runtime GPU telemetry became unavailable",), ram_violations, vram_violations
            effective_abort_temperature = min(self.policy.gpu_abort_temperature_c, gpu.max_temperature_c)
            if gpu.temperature_c >= effective_abort_temperature:
                return BenchmarkOutcome.THERMAL_LIMIT, (f"GPU {gpu.index} reached the effective abort temperature of {effective_abort_temperature} C",), ram_violations, vram_violations
            if gpu.temperature_c >= self.policy.gpu_warning_temperature_c:
                LOGGER.warning("GPU %s reached benchmark warning temperature: %s C", gpu.index, gpu.temperature_c)
            vram_violations = vram_violations + 1 if gpu.free_vram_bytes < gpu.reserved_vram_bytes else 0
            if vram_violations >= self.policy.consecutive_resource_violation_samples:
                return BenchmarkOutcome.RESOURCE_PRESSURE, (f"GPU {gpu.index} free VRAM remained below its runtime reserve",), ram_violations, vram_violations
        return None, (), ram_violations, vram_violations

    def run(self, worker: BenchmarkWorker, request: ResourceRequest) -> BenchmarkResult:
        if self._active_process is not None:
            raise RuntimeError("this supervisor already owns an active benchmark")
        started_at = datetime.now(timezone.utc)
        worker_pid = None
        outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, ("benchmark did not start",)
        worker_result = None
        worker_started = False
        timeout_kind = None
        model_deadline = None
        process = None
        messages = None
        acquired = self.state_already_acquired
        worker_shutdown_confirmed = True
        cleanup_confirmed = True
        try:
            assessment = self.resource_manager.assess(request)
            if assessment.decision is not ResourceDecision.ALLOW:
                return BenchmarkResult(BenchmarkOutcome.RESOURCE_PRESSURE, assessment.reasons, started_at, datetime.now(timezone.utc))
            if not acquired:
                self.state_store.acquire()
                acquired = True
            messages = self.context.Queue()
            process = self.context.Process(target=_worker_entry, args=(worker, messages), name="ttea-benchmark-worker")
            process.start()
            request_deadline = time.monotonic() + self.policy.request_timeout_seconds
            self._active_process = process
            worker_pid = process.pid
            self.state_store.record_worker(worker_pid)
            ram_violations = vram_violations = 0
            while True:
                try:
                    status, detail = messages.get_nowait()
                except queue.Empty:
                    status = None
                if status == "started":
                    if not worker_started:
                        worker_started = True
                        model_deadline = time.monotonic() + self.policy.model_timeout_seconds
                        if self.worker_started_callback is not None:
                            self.worker_started_callback()
                    continue
                if status == "completed":
                    if worker_started:
                        outcome, reasons, worker_result = BenchmarkOutcome.COMPLETED, ("benchmark worker completed",), detail
                    else:
                        outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, ("worker completed before the required started message",)
                    break
                if status == "failure":
                    outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, (detail,)
                    break
                if status is not None:
                    outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, (f"unknown worker message: {status!r}",)
                    break
                if not process.is_alive():
                    process.join(timeout=0)
                    try:
                        status, detail = messages.get(timeout=0.1)
                    except queue.Empty:
                        outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, (f"worker exited with code {process.exitcode}",)
                        break
                    if status == "started":
                        if not worker_started:
                            worker_started = True
                            model_deadline = time.monotonic() + self.policy.model_timeout_seconds
                            if self.worker_started_callback is not None:
                                self.worker_started_callback()
                        continue
                    if status == "completed" and worker_started:
                        outcome, reasons, worker_result = BenchmarkOutcome.COMPLETED, ("benchmark worker completed",), detail
                    elif status == "failure":
                        outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, (detail,)
                    else:
                        outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, (f"invalid worker message after exit: {status!r}",)
                    break
                deadline = request_deadline if model_deadline is None else min(request_deadline, model_deadline)
                if time.monotonic() >= deadline:
                    timeout_kind = "model" if model_deadline is not None and model_deadline <= request_deadline else "request"
                    reason = "model execution timeout elapsed" if timeout_kind == "model" else "benchmark request timeout elapsed"
                    outcome, reasons = BenchmarkOutcome.TIMEOUT, (reason,)
                    break
                violation, violation_reasons, ram_violations, vram_violations = self._runtime_violation(request, assessment.selected_gpu_index, ram_violations, vram_violations)
                if violation is not None:
                    outcome, reasons = violation, violation_reasons
                    break
                time.sleep(min(self.policy.telemetry_interval_seconds, max(0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            outcome, reasons = BenchmarkOutcome.CANCELLED, ("benchmark cancelled by operator",)
        except Exception as error:
            outcome, reasons = BenchmarkOutcome.WORKER_FAILURE, (f"supervisor failure: {type(error).__name__}: {error}",)
        finally:
            if process is not None:
                try:
                    self._stop_owned_worker(process)
                except WorkerTerminationError as error:
                    worker_shutdown_confirmed = False
                    outcome = BenchmarkOutcome.WORKER_FAILURE
                    reasons = reasons + (f"worker termination could not be confirmed: {error}",)
            if messages is not None:
                messages.close()
                messages.cancel_join_thread()
            self._active_process = None if worker_shutdown_confirmed else process
            try:
                self.cleanup.run(self.policy.cooldown_timeout_seconds)
            except Exception as error:
                cleanup_confirmed = False
                LOGGER.exception("benchmark cleanup failed: %s", error)
                reasons = reasons + (f"cleanup failed: {error}",)
                if outcome is BenchmarkOutcome.COMPLETED:
                    outcome = BenchmarkOutcome.WORKER_FAILURE
            if acquired and worker_shutdown_confirmed and cleanup_confirmed:
                try:
                    self.state_store.release()
                except Exception as error:
                    outcome = BenchmarkOutcome.WORKER_FAILURE
                    reasons = reasons + (f"ownership state release failed: {error}",)
        if outcome is not BenchmarkOutcome.TIMEOUT:
            timeout_kind = None
        return BenchmarkResult(outcome, reasons, started_at, datetime.now(timezone.utc), worker_pid, worker_result,
                               worker_started, timeout_kind)
