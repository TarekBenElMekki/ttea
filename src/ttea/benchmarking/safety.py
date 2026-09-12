"""Idempotent cleanup and verified benchmark PID-state ownership."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading

import psutil


class CleanupFailure(RuntimeError):
    """One or more bounded cleanup steps failed or timed out."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class IdempotentCleanup:
    """Run ordered cleanup steps once, bounding each in a daemon helper thread."""

    def __init__(self, callbacks: tuple[Callable[[], None], ...] = ()) -> None:
        self._callbacks = callbacks
        self._lock = threading.Lock()
        self._done = False

    @staticmethod
    def _invoke(callback: Callable[[], None], errors: list[Exception]) -> None:
        try:
            callback()
        except Exception as error:
            errors.append(error)

    def run(self, step_timeout_seconds: float = 10.0) -> None:
        if not isinstance(step_timeout_seconds, (int, float)) or isinstance(step_timeout_seconds, bool) or not math.isfinite(step_timeout_seconds) or step_timeout_seconds <= 0:
            raise ValueError("step_timeout_seconds must be a finite positive number")
        with self._lock:
            if self._done:
                return
            self._done = True
        failures: list[str] = []
        for index, callback in enumerate(self._callbacks, start=1):
            errors: list[Exception] = []
            thread = threading.Thread(target=self._invoke, args=(callback, errors), daemon=True, name=f"ttea-cleanup-{index}")
            thread.start()
            thread.join(timeout=step_timeout_seconds)
            if thread.is_alive():
                failures.append(f"cleanup step {index} timed out after {step_timeout_seconds} seconds")
            elif errors:
                failures.append(f"cleanup step {index} failed: {type(errors[0]).__name__}: {errors[0]}")
        if failures:
            raise CleanupFailure(tuple(failures))


class BenchmarkStateStore:
    """Maintain minimal state used only to identify supervisor-owned children."""

    def __init__(self, path: Path = Path("data/runtime/benchmark.json")) -> None:
        self.path = path

    @staticmethod
    def process_matches(pid: object, started_epoch: object) -> bool:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not isinstance(started_epoch, (int, float)):
            return False
        try:
            return abs(psutil.Process(pid).create_time() - float(started_epoch)) <= 2.0
        except (psutil.Error, OSError, ValueError):
            return False

    def _read(self) -> dict[str, object] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            state = self._read()
            if state and self.process_matches(state.get("supervisor_pid"), state.get("supervisor_started_epoch")):
                raise RuntimeError("another recorded benchmark supervisor is still active")
            if state and self.process_matches(state.get("worker_pid"), state.get("worker_started_epoch")):
                raise RuntimeError("another recorded benchmark worker is still active")
            if state and self.process_matches(state.get("server_pid"), state.get("server_started_epoch")):
                raise RuntimeError("another recorded Ollama server is still active")
            self.path.unlink(missing_ok=True)
        supervisor_started = psutil.Process(os.getpid()).create_time()
        state = {"supervisor_pid": os.getpid(), "supervisor_started_epoch": supervisor_started,
                 "server_pid": None, "server_started_epoch": None,
                 "worker_pid": None, "worker_started_epoch": None,
                 "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            with self.path.open("x", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True)
        except FileExistsError as error:
            raise RuntimeError("benchmark state was acquired concurrently") from error

    def record_worker(self, pid: int) -> None:
        process_started = psutil.Process(pid).create_time()
        state = self._read() or {}
        if state.get("supervisor_pid") != os.getpid() or not self.process_matches(state.get("supervisor_pid"), state.get("supervisor_started_epoch")):
            raise RuntimeError("benchmark state is not owned by this supervisor")
        state.update(worker_pid=pid, worker_started_epoch=process_started)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def record_server(self, pid: int, started_epoch: float) -> None:
        if not self.process_matches(pid, started_epoch):
            raise RuntimeError("owned Ollama server identity could not be verified")
        state = self._read() or {}
        if state.get("supervisor_pid") != os.getpid() or not self.process_matches(state.get("supervisor_pid"), state.get("supervisor_started_epoch")):
            raise RuntimeError("benchmark state is not owned by this supervisor")
        state.update(server_pid=pid, server_started_epoch=started_epoch)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def release(self) -> None:
        state = self._read()
        if state and state.get("supervisor_pid") == os.getpid() and self.process_matches(state.get("supervisor_pid"), state.get("supervisor_started_epoch")):
            if self.process_matches(state.get("worker_pid"), state.get("worker_started_epoch")):
                raise RuntimeError("cannot release state while the recorded worker is active")
            if self.process_matches(state.get("server_pid"), state.get("server_started_epoch")):
                raise RuntimeError("cannot release state while the recorded Ollama server is active")
            self.path.unlink(missing_ok=True)
