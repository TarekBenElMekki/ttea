"""Fail-safe supervision primitives for future local-model benchmarks."""

from .models import BenchmarkOutcome, BenchmarkResult, BenchmarkSafetyPolicy
from .safety import BenchmarkStateStore, CleanupFailure, IdempotentCleanup
from .supervisor import BenchmarkSupervisor, BenchmarkWorker, FailingBenchmarkWorker, NoOpBenchmarkWorker, SleepingBenchmarkWorker, WorkerTerminationError

__all__ = ["BenchmarkOutcome", "BenchmarkResult", "BenchmarkSafetyPolicy", "BenchmarkStateStore", "BenchmarkSupervisor", "BenchmarkWorker", "CleanupFailure", "FailingBenchmarkWorker", "IdempotentCleanup", "NoOpBenchmarkWorker", "SleepingBenchmarkWorker", "WorkerTerminationError"]
