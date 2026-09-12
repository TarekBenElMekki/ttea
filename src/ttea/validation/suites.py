"""Registered real-system validation suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import psutil

from ttea.benchmarking import BenchmarkSafetyPolicy, BenchmarkStateStore, OllamaClient, OllamaEndpoint, OwnedOllamaServer, resolve_ollama_executable
from ttea.benchmarking.ollama import OllamaRuntimeError, OwnedProcessIdentity
from ttea.resources import ResourceDecision, ResourceManager, ResourceRequest
from ttea.telemetry import HardwareTelemetryService

from .models import ValidationStatus
from .runner import SuiteRegistry, ValidationSuite
from .scenario import ScenarioCheck, ScenarioVerification, ValidationScenario


DEFAULT_ENDPOINT = OllamaEndpoint("http://127.0.0.1:11434")
DEDICATED_ENDPOINT = OllamaEndpoint("http://127.0.0.1:11435")
MIB = 1024 * 1024
MISSING_MODEL = "__ttea_nonexistent_smoke_model__"


def loopback_port_open(port: int, timeout_seconds: float = 0.25) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def wait_for_port_closed(port: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not loopback_port_open(port):
            return True
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return not loopback_port_open(port)


def listening_identity(port: int) -> OwnedProcessIdentity | None:
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None
    for connection in connections:
        if connection.status != psutil.CONN_LISTEN or connection.pid is None or not connection.laddr:
            continue
        if connection.laddr.port != port:
            continue
        try:
            process = psutil.Process(connection.pid)
            return OwnedProcessIdentity(connection.pid, process.create_time())
        except (psutil.Error, OSError):
            return None
    return None


def stop_verified_process(identity: OwnedProcessIdentity, timeout_seconds: float = 3.0, process_api=psutil.Process) -> None:
    try:
        process = process_api(identity.pid)
    except psutil.NoSuchProcess:
        return
    if process.pid != identity.pid or abs(process.create_time() - identity.started_epoch) > 2.0:
        raise RuntimeError("owned process identity no longer matches")
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds / 2)
    except psutil.NoSuchProcess:
        return
    except psutil.TimeoutExpired:
        if process.pid != identity.pid or abs(process.create_time() - identity.started_epoch) > 2.0:
            raise RuntimeError("owned process identity changed before forced stop")
        process.kill()
        try:
            process.wait(timeout=timeout_seconds / 2)
        except psutil.NoSuchProcess:
            return


@dataclass
class PreModelContext:
    project_path: Path = field(default_factory=Path.cwd)
    state_path: Path = Path("data/runtime/benchmark.json")
    telemetry: HardwareTelemetryService | None = None
    resource_manager: ResourceManager | None = None
    benchmark_policy: BenchmarkSafetyPolicy = field(default_factory=BenchmarkSafetyPolicy)
    executable: str | None = None
    default_port_was_open: bool = False
    default_identity: OwnedProcessIdentity | None = None
    dedicated_port_was_open: bool = False
    dedicated_identity: OwnedProcessIdentity | None = None
    baseline_budget: Any = None
    baseline_telemetry_error: str | None = None
    server: OwnedOllamaServer | None = None
    child: subprocess.Popen | None = None
    child_identity: OwnedProcessIdentity | None = None
    state_store: BenchmarkStateStore | None = None
    ram_recovery_min_bytes: int = 256 * MIB
    vram_recovery_min_bytes: int = 64 * MIB
    recovery_fraction: float = 0.05
    settle_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.telemetry is None:
            self.telemetry = HardwareTelemetryService(self.project_path)
        if self.resource_manager is None:
            self.resource_manager = ResourceManager(self.telemetry)
        if self.state_store is None:
            self.state_store = BenchmarkStateStore(self.state_path)

    def server_resource_check(self) -> ScenarioCheck:
        if self.dedicated_port_was_open or loopback_port_open(11435):
            return ScenarioCheck(ValidationStatus.BLOCKED, ("dedicated loopback port 11435 is occupied; no process was terminated",))
        request = ResourceRequest("validation-model-less-server", 256 * MIB, 16 * MIB)
        try:
            assessment = self.resource_manager.assess(request)
        except Exception as error_value:
            return ScenarioCheck(ValidationStatus.BLOCKED, (f"resource telemetry unavailable: {error_value}",))
        if assessment.decision is not ResourceDecision.ALLOW:
            return ScenarioCheck(ValidationStatus.BLOCKED, assessment.reasons,
                                 {"resource_decision": assessment.decision.value},
                                 assessment.decision is ResourceDecision.DELAY)
        return ScenarioCheck(evidence={"resource_decision": assessment.decision.value})

    def start_server(self) -> dict[str, Any]:
        if self.server is not None:
            raise RuntimeError("validation context already owns a server")
        server = OwnedOllamaServer(
            DEDICATED_ENDPOINT, self.benchmark_policy, executable=self.executable,
            identity_callback=lambda identity: self.state_store.record_server(identity.pid, identity.started_epoch),
        )
        self.server = server
        startup = server.start()
        if not startup.success or startup.identity is None:
            raise RuntimeError("; ".join(startup.reasons))
        client = server.client()
        version = client.version()
        budget = self.resource_manager.current_budget()
        if budget.available_ram_bytes < budget.reserved_ram_bytes:
            raise RuntimeError("runtime RAM fell below the configured reserve")
        if listening_identity(11435) != startup.identity:
            raise RuntimeError("dedicated listener identity does not match the owned server")
        return {"server_pid": startup.identity.pid, "server_started_epoch": startup.identity.started_epoch,
                "api_version": version, "port": 11435}

    def stop_server(self) -> None:
        if self.server is None:
            return
        server = self.server
        server.shutdown(self.benchmark_policy.cooldown_timeout_seconds)
        if not server.shutdown_confirmed:
            raise RuntimeError("owned Ollama server shutdown was not confirmed")
        if not wait_for_port_closed(11435):
            raise RuntimeError("dedicated port 11435 remains open after server shutdown")
        self.server = None


class PreModelRunGuard:
    def __init__(self, context: PreModelContext) -> None:
        self.context = context
        self.environment_summary: dict[str, Any] = {}

    def establish(self) -> None:
        self.context.state_store.acquire()
        try:
            self.context.default_port_was_open = loopback_port_open(11434)
            self.context.default_identity = listening_identity(11434) if self.context.default_port_was_open else None
            self.context.dedicated_port_was_open = loopback_port_open(11435)
            self.context.dedicated_identity = listening_identity(11435) if self.context.dedicated_port_was_open else None
            try:
                self.context.baseline_budget = self.context.resource_manager.current_budget()
            except Exception as error_value:
                self.context.baseline_telemetry_error = str(error_value)
            self.environment_summary = {
                "default_ollama_present_before": self.context.default_port_was_open,
                "default_ollama_pid_before": None if self.context.default_identity is None else self.context.default_identity.pid,
                "dedicated_port_open_before": self.context.dedicated_port_was_open,
                "dedicated_port_pid_before": None if self.context.dedicated_identity is None else self.context.dedicated_identity.pid,
            }
        except BaseException:
            self.context.state_store.release()
            raise

    def finalize(self) -> None:
        errors = []
        try:
            self.context.stop_server()
        except BaseException as error_value:
            errors.append(str(error_value))
        if self.context.child_identity is not None:
            try:
                stop_verified_process(self.context.child_identity)
                self.context.child_identity = None
                self.context.child = None
            except BaseException as error_value:
                errors.append(f"owned validation child remains: {error_value}")
        state = self.context.state_store.snapshot() or {}
        for kind in ("worker", "server"):
            if self.context.state_store.process_matches(state.get(f"{kind}_pid"), state.get(f"{kind}_started_epoch")):
                errors.append(f"recorded owned {kind} remains active")
        if self.context.dedicated_port_was_open:
            if self.context.dedicated_identity is not None and listening_identity(11435) != self.context.dedicated_identity:
                errors.append("pre-existing dedicated-port listener identity changed")
        elif not wait_for_port_closed(11435):
            errors.append("dedicated port 11435 remains open")
        if self.context.default_identity is not None:
            current = listening_identity(11434)
            if current != self.context.default_identity:
                errors.append("default Ollama listener identity changed")
        if errors:
            raise RuntimeError("; ".join(errors))
        self.context.state_store.release()
        if self.context.state_path.exists():
            raise RuntimeError("benchmark runtime state remains after verified cleanup")


class EnvironmentScenario(ValidationScenario):
    scenario_id = "environment"

    def execute(self, context: PreModelContext):
        executable = resolve_ollama_executable(context.executable)
        if executable is None:
            raise RuntimeError("Ollama executable could not be resolved")
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=5.0, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Ollama version command failed with code {result.returncode}")
        version = (result.stdout or result.stderr).strip()
        if not version:
            raise RuntimeError("Ollama version command returned no version")
        context.executable = executable
        return {"python_prefix": sys.prefix, "project_path": str(context.project_path.resolve()),
                "project_writable": os.access(context.project_path, os.W_OK), "ollama_executable": executable,
                "ollama_version": version}

    def verify(self, context, execution):
        if not execution["project_writable"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("repository path is not writable",))
        return ScenarioVerification(reasons=("Python environment, repository path, and Ollama executable are valid",))


class PassiveDefaultOllamaScenario(ValidationScenario):
    scenario_id = "passive_default_ollama"

    def execute(self, context: PreModelContext):
        if not context.default_port_was_open:
            return {"present": False, "models": []}
        client = OllamaClient(DEFAULT_ENDPOINT, context.benchmark_policy, 2.0)
        return {"present": True, "version": client.version(), "models": list(client.installed_models()),
                "pid": None if context.default_identity is None else context.default_identity.pid,
                "started_epoch": None if context.default_identity is None else context.default_identity.started_epoch}

    def verify(self, context, execution):
        reason = "default Ollama API recorded without modification" if execution["present"] else "default Ollama API was not present"
        return ScenarioVerification(reasons=(reason,))


class ResourcePreflightScenario(ValidationScenario):
    scenario_id = "resource_preflight"

    def precheck(self, context):
        if context.baseline_budget is None:
            return ScenarioCheck(ValidationStatus.BLOCKED, (f"baseline telemetry unavailable: {context.baseline_telemetry_error}",))
        return ScenarioCheck()

    def execute(self, context):
        budget = context.baseline_budget
        consistent = budget.allocatable_ram_bytes <= budget.available_ram_bytes and budget.allocatable_disk_bytes <= budget.free_disk_bytes
        if any(gpu.allocatable_vram_bytes > (gpu.free_vram_bytes or 0) for gpu in budget.gpus):
            consistent = False
        assessment = context.resource_manager.assess(ResourceRequest("validation-model-less-server", 256 * MIB, 16 * MIB))
        return {"snapshot_timestamp": budget.snapshot_timestamp.isoformat(), "ram_available_bytes": budget.available_ram_bytes,
                "ram_reserved_bytes": budget.reserved_ram_bytes, "ram_allocatable_bytes": budget.allocatable_ram_bytes,
                "disk_free_bytes": budget.free_disk_bytes, "disk_reserved_bytes": budget.reserved_disk_bytes,
                "telemetry_consistent": consistent, "resource_decision": assessment.decision.value,
                "resource_reasons": list(assessment.reasons)}

    def verify(self, context, execution):
        if not execution["telemetry_consistent"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("resource telemetry is internally inconsistent",))
        if execution["resource_decision"] != ResourceDecision.ALLOW.value:
            return ScenarioVerification(ValidationStatus.BLOCKED, tuple(execution["resource_reasons"]))
        return ScenarioVerification(reasons=("resource telemetry is internally consistent",))


class DedicatedPortPrecheckScenario(ValidationScenario):
    scenario_id = "dedicated_port_precheck"

    def precheck(self, context):
        if context.dedicated_port_was_open:
            identity = listening_identity(11435)
            return ScenarioCheck(ValidationStatus.BLOCKED, ("dedicated port 11435 was already occupied; no process was terminated",),
                                 {"listener_pid": None if identity is None else identity.pid})
        return ScenarioCheck()

    def verify(self, context, execution):
        return ScenarioVerification(reasons=("dedicated loopback port 11435 is available",), evidence={"port": 11435})


class OwnedServerLifecycleScenario(ValidationScenario):
    scenario_id = "owned_server_lifecycle"

    def precheck(self, context):
        return context.server_resource_check()

    def execute(self, context):
        return context.start_server()

    def verify(self, context, execution):
        return ScenarioVerification(reasons=("owned model-less Ollama server became healthy with verified identity",))

    def cleanup(self, context):
        context.stop_server()


class MissingModelRefusalScenario(OwnedServerLifecycleScenario):
    scenario_id = "missing_model_refusal"

    def execute(self, context):
        evidence = context.start_server()
        client = context.server.client()
        installed = client.installed_models()
        evidence.update({"model": MISSING_MODEL, "installed": MISSING_MODEL in installed,
                         "automatic_pull_attempted": False, "generation_worker_started": False,
                         "model_cleanup_attempted": False})
        return evidence

    def verify(self, context, execution):
        if execution["installed"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("reserved fake model unexpectedly exists locally",))
        return ScenarioVerification(reasons=("missing model was refused before load or generation; server-only cleanup applies",))


class DefaultServerIsolationScenario(ValidationScenario):
    scenario_id = "default_server_isolation"

    def precheck(self, context):
        if not context.default_port_was_open:
            return ScenarioCheck(ValidationStatus.SKIP, ("default Ollama server was absent before validation",))
        if context.default_identity is None:
            return ScenarioCheck(ValidationStatus.BLOCKED, ("default Ollama listener PID identity could not be resolved safely",))
        return ScenarioCheck()

    def execute(self, context):
        current = listening_identity(11434)
        return {"pid_before": context.default_identity.pid, "pid_after": None if current is None else current.pid,
                "identity_preserved": current == context.default_identity}

    def verify(self, context, execution):
        status = ValidationStatus.PASS if execution["identity_preserved"] else ValidationStatus.FAIL
        reason = "default Ollama process identity was preserved" if status is ValidationStatus.PASS else "default Ollama process identity changed"
        return ScenarioVerification(status, (reason,))


class ResourceRecoveryScenario(ValidationScenario):
    scenario_id = "resource_recovery"

    def precheck(self, context):
        if context.baseline_budget is None:
            return ScenarioCheck(ValidationStatus.BLOCKED, ("baseline resource telemetry is unavailable",))
        return ScenarioCheck()

    def execute(self, context):
        time.sleep(context.settle_seconds)
        after = context.resource_manager.current_budget()
        before = context.baseline_budget
        ram_delta = after.available_ram_bytes - before.available_ram_bytes
        ram_tolerance = max(context.ram_recovery_min_bytes, int(before.available_ram_bytes * context.recovery_fraction))
        before_gpus = {gpu.index: gpu for gpu in before.gpus}
        gpu_deltas = []
        for gpu in after.gpus:
            previous = before_gpus.get(gpu.index)
            if previous is None or previous.free_vram_bytes is None or gpu.free_vram_bytes is None:
                continue
            tolerance = max(context.vram_recovery_min_bytes, int(previous.free_vram_bytes * context.recovery_fraction))
            gpu_deltas.append({"index": gpu.index, "free_vram_delta_bytes": gpu.free_vram_bytes - previous.free_vram_bytes,
                               "tolerance_bytes": tolerance, "temperature_before_c": previous.temperature_c,
                               "temperature_after_c": gpu.temperature_c})
        return {"ram_available_before_bytes": before.available_ram_bytes, "ram_available_after_bytes": after.available_ram_bytes,
                "ram_delta_bytes": ram_delta, "ram_tolerance_bytes": ram_tolerance, "gpu_deltas": gpu_deltas,
                "recovered": ram_delta >= -ram_tolerance and all(item["free_vram_delta_bytes"] >= -item["tolerance_bytes"] for item in gpu_deltas)}

    def verify(self, context, execution):
        if not execution["recovered"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("resource use did not recover within configured tolerance",))
        return ScenarioVerification(reasons=("RAM and VRAM recovered within configured OS-noise tolerances",))


class EmergencyRecoveryContractScenario(ValidationScenario):
    scenario_id = "emergency_recovery_contract"

    def execute(self, context):
        kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        context.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        created = psutil.Process(context.child.pid).create_time()
        context.child_identity = OwnedProcessIdentity(context.child.pid, created)
        context.state_store.record_worker(context.child.pid)
        unrelated = OwnedProcessIdentity(os.getpid(), psutil.Process(os.getpid()).create_time())
        stop_verified_process(context.child_identity)
        context.child.wait(timeout=1.0)
        stopped_pid = context.child_identity.pid
        context.child_identity = None
        context.child = None
        return {"owned_child_pid": stopped_pid, "owned_child_stopped": True,
                "unrelated_pid": unrelated.pid, "unrelated_identity_preserved": BenchmarkStateStore.process_matches(unrelated.pid, unrelated.started_epoch)}

    def verify(self, context, execution):
        if not execution["unrelated_identity_preserved"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("unrelated process identity was not preserved",))
        return ScenarioVerification(reasons=("exact owned child stopped and unrelated process identity remained untouched",))

    def cleanup(self, context):
        if context.child_identity is not None:
            stop_verified_process(context.child_identity)
            context.child_identity = None
            context.child = None


def build_registry() -> SuiteRegistry:
    registry = SuiteRegistry()
    registry.register(ValidationSuite(
        "pre_model",
        (EnvironmentScenario, PassiveDefaultOllamaScenario, ResourcePreflightScenario,
         DedicatedPortPrecheckScenario, OwnedServerLifecycleScenario, MissingModelRefusalScenario,
         DefaultServerIsolationScenario, ResourceRecoveryScenario, EmergencyRecoveryContractScenario),
        PreModelContext,
        PreModelRunGuard,
    ))
    return registry
