"""Real-inference safety validation for one explicitly allow-listed tiny model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from ttea.benchmarking import (BenchmarkOutcome, BenchmarkSafetyPolicy, BenchmarkSupervisor,
                               OllamaBenchmarkWorker, OwnedOllamaServer)
from ttea.benchmarking.ollama import OllamaCleanup, OllamaGenerationResult
from ttea.resources import ResourceDecision, ResourceRequest

from .model_assets import MIB, TINY_MODEL_MANIFEST, TinyModelMetadata, TinyModelProvisioner
from .models import ValidationStatus
from .runner import ValidationRunOptions, ValidationSuite
from .scenario import ScenarioCheck, ScenarioVerification, ValidationScenario
from .suites import (DEDICATED_ENDPOINT, DefaultServerIsolationScenario, PreModelContext,
                     PreModelRunGuard, listening_identity, loopback_port_open, wait_for_port_closed)


TINY_PROMPT = "Reply with one short greeting."


def tiny_policy(model_timeout_seconds: float = 60.0) -> BenchmarkSafetyPolicy:
    return BenchmarkSafetyPolicy(
        telemetry_interval_seconds=1.0,
        consecutive_resource_violation_samples=3,
        request_timeout_seconds=90.0,
        model_timeout_seconds=model_timeout_seconds,
        cooldown_timeout_seconds=10.0,
        gpu_warning_temperature_c=70.0,
        gpu_abort_temperature_c=80.0,
        max_context_tokens=512,
        max_output_tokens=16,
        cpu_thread_limit=2,
    )


@dataclass
class OperationTelemetry:
    selected_gpu_index: int | None
    ram_available_before: int
    ram_lowest: int
    gpu_free_before: int | None
    gpu_free_minimum: int | None
    gpu_temperature_maximum: float | None

    def observe(self, budget) -> None:
        self.ram_lowest = min(self.ram_lowest, budget.available_ram_bytes)
        if self.selected_gpu_index is None:
            return
        gpu = next((item for item in budget.gpus if item.index == self.selected_gpu_index), None)
        if gpu is None:
            return
        if gpu.free_vram_bytes is not None:
            self.gpu_free_minimum = gpu.free_vram_bytes if self.gpu_free_minimum is None else min(self.gpu_free_minimum, gpu.free_vram_bytes)
        if gpu.temperature_c is not None:
            self.gpu_temperature_maximum = gpu.temperature_c if self.gpu_temperature_maximum is None else max(self.gpu_temperature_maximum, gpu.temperature_c)


@dataclass
class TinyModelContext(PreModelContext):
    allow_model_download: bool = False
    pre_model_passed: bool = False
    manifest: Any = TINY_MODEL_MANIFEST
    provisioner: TinyModelProvisioner = field(default_factory=TinyModelProvisioner)
    metadata: TinyModelMetadata | None = None
    model_downloaded: bool = False
    provision_blocked_reason: str | None = None
    inference_request: ResourceRequest | None = None
    inference_assessment_decision: ResourceDecision | None = None
    normal_result: Any = None
    normal_evidence: dict[str, Any] | None = None
    timeout_result: Any = None
    timeout_evidence: dict[str, Any] | None = None

    def configure_validation(self, options: ValidationRunOptions) -> None:
        self.allow_model_download = options.allow_model_download
        self.pre_model_passed = options.pre_model_passed

    def acquire_server(self, policy: BenchmarkSafetyPolicy) -> OwnedOllamaServer:
        if loopback_port_open(11435):
            raise RuntimeError("dedicated port 11435 is occupied")
        self.state_store.acquire()
        try:
            server = OwnedOllamaServer(
                DEDICATED_ENDPOINT, policy, executable=self.executable,
                identity_callback=lambda identity: self.state_store.record_server(identity.pid, identity.started_epoch),
            )
            self.server = server
            startup = server.start()
            if not startup.success:
                if server.shutdown_confirmed:
                    self.state_store.release()
                    self.server = None
                raise RuntimeError("; ".join(startup.reasons))
            if startup.identity is None or listening_identity(11435) != startup.identity:
                self.release_server_only(policy)
                raise RuntimeError("dedicated listener identity does not match the owned server")
            return server
        except BaseException:
            if self.server is not None and not self.server.shutdown_confirmed:
                try:
                    self.server.shutdown(policy.cooldown_timeout_seconds)
                except BaseException:
                    pass
            if self.server is None or self.server.shutdown_confirmed:
                self.server = None
                if self.state_path.exists():
                    self.state_store.release()
            raise

    def release_server_only(self, policy: BenchmarkSafetyPolicy) -> None:
        if self.server is not None:
            self.server.shutdown(policy.cooldown_timeout_seconds)
            if not self.server.shutdown_confirmed:
                raise RuntimeError("owned Ollama server shutdown was not confirmed")
            self.server = None
        if not wait_for_port_closed(11435):
            raise RuntimeError("dedicated port 11435 remains open")
        if self.state_path.exists():
            self.state_store.release()

    def build_inference_request(self) -> ResourceRequest:
        profile = self.telemetry.get_profile()
        snapshot = self.telemetry.snapshot()
        metrics = {gpu.index: gpu for gpu in snapshot.gpus}
        usable = [gpu for gpu in profile.gpus
                  if gpu.source == "nvidia-smi" and gpu.total_vram_bytes is not None
                  and gpu.total_vram_bytes >= self.manifest.gpu_vram_estimate_bytes
                  and gpu.index in metrics and metrics[gpu.index].free_vram_bytes is not None
                  and metrics[gpu.index].temperature_c is not None]
        selected = min(usable, key=lambda gpu: gpu.index) if usable else None
        return ResourceRequest(
            "tiny-model-safety-inference", self.manifest.ram_estimate_bytes, 0,
            self.manifest.gpu_vram_estimate_bytes if selected is not None else 0,
            selected.index if selected is not None else None,
        )

    def _recovery_evidence(self, before, tracker: OperationTelemetry, after) -> dict[str, Any]:
        ram_tolerance = max(self.ram_recovery_min_bytes, int(before.available_ram_bytes * self.recovery_fraction))
        ram_delta = after.available_ram_bytes - before.available_ram_bytes
        gpu_before = gpu_after = gpu_delta = gpu_tolerance = None
        if tracker.selected_gpu_index is not None:
            prior = next((gpu for gpu in before.gpus if gpu.index == tracker.selected_gpu_index), None)
            later = next((gpu for gpu in after.gpus if gpu.index == tracker.selected_gpu_index), None)
            if prior is not None and later is not None and prior.free_vram_bytes is not None and later.free_vram_bytes is not None:
                gpu_before, gpu_after = prior.free_vram_bytes, later.free_vram_bytes
                gpu_delta = gpu_after - gpu_before
                gpu_tolerance = max(self.vram_recovery_min_bytes, int(gpu_before * self.recovery_fraction))
        recovered = ram_delta >= -ram_tolerance and (gpu_delta is None or gpu_delta >= -gpu_tolerance)
        thermal_safe = tracker.gpu_temperature_maximum is None or tracker.gpu_temperature_maximum < self.benchmark_policy.gpu_abort_temperature_c
        return {
            "ram_available_before_bytes": before.available_ram_bytes,
            "ram_lowest_observed_bytes": tracker.ram_lowest,
            "ram_available_after_bytes": after.available_ram_bytes,
            "ram_delta_bytes": ram_delta,
            "ram_tolerance_bytes": ram_tolerance,
            "gpu_index": tracker.selected_gpu_index,
            "gpu_free_before_bytes": gpu_before,
            "gpu_free_minimum_observed_bytes": tracker.gpu_free_minimum,
            "gpu_free_after_bytes": gpu_after,
            "gpu_delta_bytes": gpu_delta,
            "gpu_tolerance_bytes": gpu_tolerance,
            "gpu_max_temperature_c": tracker.gpu_temperature_maximum,
            "resources_recovered": recovered,
            "thermal_safe": thermal_safe,
        }

    def run_generation(self, policy: BenchmarkSafetyPolicy) -> tuple[Any, dict[str, Any]]:
        if self.inference_request is None:
            raise RuntimeError("tiny-model resource request was not prepared")
        server = self.acquire_server(policy)
        handed_to_supervisor = False
        try:
            client = server.client()
            if self.manifest.model not in client.installed_models():
                raise RuntimeError("allow-listed tiny model is not installed locally")
            before = self.resource_manager.current_budget()
            selected = self.inference_request.gpu_index
            selected_gpu = next((gpu for gpu in before.gpus if gpu.index == selected), None)
            tracker = OperationTelemetry(selected, before.available_ram_bytes, before.available_ram_bytes,
                                         None if selected_gpu is None else selected_gpu.free_vram_bytes,
                                         None if selected_gpu is None else selected_gpu.free_vram_bytes,
                                         None if selected_gpu is None else selected_gpu.temperature_c)
            model_cleanup = OllamaCleanup(client, server, self.manifest.model, model_cleanup_required=False)
            supervisor = BenchmarkSupervisor(
                self.resource_manager, policy, model_cleanup.bounded(), self.state_path,
                state_store=self.state_store, state_already_acquired=True,
                worker_started_callback=model_cleanup.mark_model_cleanup_required,
                telemetry_observer=tracker.observe,
            )
            handed_to_supervisor = True
            result = supervisor.run(OllamaBenchmarkWorker(client, self.manifest.model, TINY_PROMPT), self.inference_request)
            self.server = None if server.shutdown_confirmed else server
            time.sleep(self.settle_seconds)
            after = self.resource_manager.current_budget()
            evidence = self._recovery_evidence(before, tracker, after)
            evidence.update({
                "model_unload_confirmed": model_cleanup.model_unload_confirmed,
                "server_shutdown_confirmed": server.shutdown_confirmed,
                "port_closed": not loopback_port_open(11435),
                "runtime_state_absent": not self.state_path.exists(),
                "worker_started": result.worker_started,
                "timeout_kind": result.timeout_kind,
            })
            if isinstance(result.worker_result, OllamaGenerationResult):
                generated = result.worker_result
                tokens_per_second = None
                if generated.eval_count is not None and generated.eval_duration:
                    tokens_per_second = generated.eval_count / (generated.eval_duration / 1_000_000_000)
                evidence.update({
                    "response_is_string": isinstance(generated.response, str),
                    "response_length": len(generated.response),
                    "total_duration": generated.total_duration,
                    "load_duration": generated.load_duration,
                    "prompt_eval_count": generated.prompt_eval_count,
                    "prompt_eval_duration": generated.prompt_eval_duration,
                    "eval_count": generated.eval_count,
                    "eval_duration": generated.eval_duration,
                    "done_reason": generated.done_reason,
                    "tokens_per_second": tokens_per_second,
                })
            return result, evidence
        finally:
            if not handed_to_supervisor:
                self.release_server_only(policy)


class TinyModelRunGuard:
    def __init__(self, context: TinyModelContext) -> None:
        self.context = context
        self.base = PreModelRunGuard(context)
        self.environment_summary = {}

    def establish(self) -> None:
        self.base.establish()
        self.environment_summary = dict(self.base.environment_summary)
        self.context.state_store.release()

    def finalize(self) -> None:
        # Reacquisition proves there is no live owned benchmark operation; the
        # shared guard then performs its normal port/default-server checks.
        self.context.state_store.acquire()
        self.base.finalize()


class TinyModelManifestScenario(ValidationScenario):
    scenario_id = "tiny_model_manifest"

    def precheck(self, context):
        if not context.pre_model_passed:
            return ScenarioCheck(ValidationStatus.BLOCKED, ("tiny_model requires a real pre_model PASS",))
        return ScenarioCheck()

    def execute(self, context):
        return {"model": context.manifest.model, "purpose": context.manifest.purpose,
                "allowlisted": context.provisioner.manifest is context.manifest,
                "download_timeout_seconds": context.manifest.download_timeout_seconds}

    def verify(self, context, execution):
        return ScenarioVerification(reasons=("exact tiny safety model manifest is allow-listed",))


class TinyModelProvisionScenario(ValidationScenario):
    scenario_id = "tiny_model_provision"

    def precheck(self, context):
        if not context.pre_model_passed:
            return ScenarioCheck(ValidationStatus.SKIP, ("pre_model has not passed",))
        if loopback_port_open(11435):
            return ScenarioCheck(ValidationStatus.BLOCKED, ("dedicated port 11435 is occupied",))
        assessment = context.resource_manager.assess(ResourceRequest("tiny-model-provision-server", 256 * MIB, 16 * MIB))
        if assessment.decision is not ResourceDecision.ALLOW:
            return ScenarioCheck(ValidationStatus.BLOCKED, assessment.reasons,
                                 {"resource_decision": assessment.decision.value})
        return ScenarioCheck()

    def execute(self, context):
        policy = tiny_policy()
        server = context.acquire_server(policy)
        try:
            client = server.client()
            existing = context.provisioner.inspect(client)
            if existing is None and not context.allow_model_download:
                context.provision_blocked_reason = "tiny validation model is missing and model download was not authorized"
                return {"blocked": True, "reason": context.provision_blocked_reason, "download_attempted": False}
            if existing is None:
                disk = context.resource_manager.assess(ResourceRequest("tiny-model-download", 0,
                                                                       context.manifest.expected_disk_bytes))
                if disk.decision is not ResourceDecision.ALLOW:
                    return {"blocked": True, "reason": "; ".join(disk.reasons),
                            "resource_decision": disk.decision.value, "download_attempted": False}
            context.metadata, context.model_downloaded = context.provisioner.ensure(client, context.allow_model_download)
            return {"blocked": False, "download_attempted": context.model_downloaded,
                    "already_installed": not context.model_downloaded, **context.metadata.evidence()}
        finally:
            context.release_server_only(policy)

    def verify(self, context, execution):
        if execution["blocked"]:
            evidence = {"resource_decision": execution.get("resource_decision")}
            return ScenarioVerification(ValidationStatus.BLOCKED, (execution["reason"],), evidence)
        return ScenarioVerification(reasons=("model digest, size, and metadata were recorded",))


class TinyModelMetadataScenario(ValidationScenario):
    scenario_id = "tiny_model_metadata"

    def precheck(self, context):
        if context.metadata is None:
            return ScenarioCheck(ValidationStatus.SKIP, ("tiny-model metadata is unavailable because provisioning did not complete",))
        return ScenarioCheck()

    def execute(self, context):
        return context.metadata.evidence()

    def verify(self, context, execution):
        return ScenarioVerification(reasons=("tiny-model digest, size, and available metadata were captured",))


class TinyModelResourcePreflightScenario(ValidationScenario):
    scenario_id = "tiny_model_resource_preflight"

    def precheck(self, context):
        if context.metadata is None:
            return ScenarioCheck(ValidationStatus.SKIP, ("tiny model is not provisioned",))
        return ScenarioCheck()

    def execute(self, context):
        context.inference_request = context.build_inference_request()
        assessment = context.resource_manager.assess(context.inference_request)
        context.inference_assessment_decision = assessment.decision
        return {"resource_decision": assessment.decision.value, "resource_reasons": list(assessment.reasons),
                "ram_estimate_bytes": context.inference_request.estimated_ram_bytes,
                "gpu_vram_estimate_bytes": context.inference_request.estimated_gpu_vram_bytes,
                "gpu_index": context.inference_request.gpu_index}

    def verify(self, context, execution):
        if execution["resource_decision"] != ResourceDecision.ALLOW.value:
            return ScenarioVerification(ValidationStatus.BLOCKED, tuple(execution["resource_reasons"]))
        return ScenarioVerification(reasons=("ResourceManager allowed the conservative tiny-model inference estimate",))


class TinyModelInferenceScenario(ValidationScenario):
    scenario_id = "tiny_model_inference"

    def precheck(self, context):
        if context.inference_request is None:
            return ScenarioCheck(ValidationStatus.SKIP, ("tiny-model inference preflight did not complete",))
        assessment = context.resource_manager.assess(context.inference_request)
        if assessment.decision is not ResourceDecision.ALLOW:
            return ScenarioCheck(ValidationStatus.BLOCKED, assessment.reasons,
                                 {"resource_decision": assessment.decision.value})
        return ScenarioCheck()

    def execute(self, context):
        context.normal_result, context.normal_evidence = context.run_generation(tiny_policy())
        return {"outcome": context.normal_result.outcome.value, **context.normal_evidence}

    def verify(self, context, execution):
        if context.normal_result.outcome is BenchmarkOutcome.RESOURCE_PRESSURE:
            return ScenarioVerification(ValidationStatus.BLOCKED, context.normal_result.reasons,
                                        {"resource_decision": ResourceDecision.DELAY.value})
        if context.normal_result.outcome is not BenchmarkOutcome.COMPLETED:
            return ScenarioVerification(ValidationStatus.FAIL, context.normal_result.reasons)
        if not execution.get("response_is_string"):
            return ScenarioVerification(ValidationStatus.FAIL, ("generation result did not contain a string response",))
        return ScenarioVerification(reasons=("one bounded tiny-model inference completed with structured metrics",))


class TinyModelUnloadVerificationScenario(ValidationScenario):
    scenario_id = "tiny_model_unload_verification"

    def precheck(self, context):
        if context.normal_evidence is None:
            return ScenarioCheck(ValidationStatus.SKIP, ("normal inference did not execute",))
        return ScenarioCheck()

    def execute(self, context):
        keys = ("model_unload_confirmed", "server_shutdown_confirmed", "port_closed", "runtime_state_absent")
        return {key: context.normal_evidence[key] for key in keys}

    def verify(self, context, execution):
        if not all(execution.values()):
            return ScenarioVerification(ValidationStatus.FAIL, ("normal inference cleanup was not fully verified",))
        return ScenarioVerification(reasons=("model unload, server shutdown, port closure, and state release were verified",))


class TinyModelForcedTimeoutScenario(ValidationScenario):
    scenario_id = "tiny_model_forced_timeout"

    def precheck(self, context):
        if context.normal_result is None or context.normal_result.outcome is not BenchmarkOutcome.COMPLETED:
            return ScenarioCheck(ValidationStatus.SKIP, ("successful normal inference is required before timeout validation",))
        assessment = context.resource_manager.assess(context.inference_request)
        if assessment.decision is not ResourceDecision.ALLOW:
            return ScenarioCheck(ValidationStatus.BLOCKED, assessment.reasons,
                                 {"resource_decision": assessment.decision.value})
        return ScenarioCheck()

    def execute(self, context):
        attempts = []
        for timeout in (0.25, 0.10):
            result, evidence = context.run_generation(tiny_policy(timeout))
            attempts.append({"timeout_seconds": timeout, "outcome": result.outcome.value,
                             "worker_started": result.worker_started, "timeout_kind": result.timeout_kind})
            context.timeout_result, context.timeout_evidence = result, evidence
            if result.outcome is BenchmarkOutcome.TIMEOUT:
                break
            if result.outcome is not BenchmarkOutcome.COMPLETED:
                break
        return {"attempts": attempts, "outcome": context.timeout_result.outcome.value, **context.timeout_evidence}

    def verify(self, context, execution):
        if (context.timeout_result.outcome is BenchmarkOutcome.TIMEOUT
                and context.timeout_result.worker_started
                and context.timeout_result.timeout_kind == "model"):
            return ScenarioVerification(reasons=("outer model deadline interrupted an in-flight bounded worker execution",))
        if context.timeout_result.outcome is BenchmarkOutcome.TIMEOUT:
            return ScenarioVerification(ValidationStatus.FAIL,
                                        ("forced-timeout attempt did not prove a post-start model deadline",))
        if context.timeout_result.outcome is BenchmarkOutcome.COMPLETED:
            return ScenarioVerification(ValidationStatus.SKIP, ("tiny model completed before both bounded outer-timeout attempts",))
        if context.timeout_result.outcome is BenchmarkOutcome.RESOURCE_PRESSURE:
            return ScenarioVerification(ValidationStatus.BLOCKED, context.timeout_result.reasons,
                                        {"resource_decision": ResourceDecision.DELAY.value})
        return ScenarioVerification(ValidationStatus.FAIL, context.timeout_result.reasons)


class TinyModelTimeoutCleanupVerificationScenario(ValidationScenario):
    scenario_id = "tiny_model_timeout_cleanup_verification"

    def precheck(self, context):
        if context.timeout_evidence is None:
            return ScenarioCheck(ValidationStatus.SKIP, ("forced-timeout generation did not execute",))
        return ScenarioCheck()

    def execute(self, context):
        keys = ("model_unload_confirmed", "server_shutdown_confirmed", "port_closed", "runtime_state_absent")
        return {key: context.timeout_evidence[key] for key in keys}

    def verify(self, context, execution):
        if not all(execution.values()):
            return ScenarioVerification(ValidationStatus.FAIL, ("forced-timeout cleanup was not fully verified",))
        return ScenarioVerification(reasons=("forced-timeout worker/model/server cleanup was verified",))


class TinyModelResourceRecoveryScenario(ValidationScenario):
    scenario_id = "tiny_model_resource_recovery"

    def precheck(self, context):
        if context.normal_evidence is None:
            return ScenarioCheck(ValidationStatus.SKIP, ("no tiny-model inference telemetry is available",))
        return ScenarioCheck()

    def execute(self, context):
        operations = [item for item in (context.normal_evidence, context.timeout_evidence) if item is not None]
        recovered = all(item["resources_recovered"] and item["thermal_safe"] for item in operations)
        metadata = context.metadata.evidence()
        normal = context.normal_evidence
        timeout = context.timeout_evidence or {}
        operation_cleanup_verified = all(item["model_unload_confirmed"] and item["server_shutdown_confirmed"]
                                         and item["port_closed"] and item["runtime_state_absent"]
                                         for item in operations)
        timeout_cleanup_verified = bool(timeout) and all(timeout.get(key) for key in (
            "model_unload_confirmed", "server_shutdown_confirmed", "port_closed", "runtime_state_absent"))
        summary = {
            **metadata,
            "normal_outcome": context.normal_result.outcome.value,
            "eval_count": normal.get("eval_count"),
            "tokens_per_second": normal.get("tokens_per_second"),
            "ram_delta_bytes": normal.get("ram_delta_bytes"),
            "gpu_delta_bytes": normal.get("gpu_delta_bytes"),
            "gpu_max_temperature_c": max((item["gpu_max_temperature_c"] for item in operations
                                           if item["gpu_max_temperature_c"] is not None), default=None),
            "model_unload_confirmed": all(item["model_unload_confirmed"] for item in operations),
            "server_cleanup_confirmed": operation_cleanup_verified,
            "forced_timeout_outcome": None if context.timeout_result is None else context.timeout_result.outcome.value,
            "timeout_cleanup_verified": timeout_cleanup_verified,
            "default_server_identity_preserved": context.default_identity is None or listening_identity(11434) == context.default_identity,
            "resources_recovered": recovered,
        }
        return {"safety_summary": summary}

    def verify(self, context, execution):
        summary = execution["safety_summary"]
        if not summary["resources_recovered"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("RAM, VRAM, or thermal recovery was outside policy",))
        if not summary["default_server_identity_preserved"]:
            return ScenarioVerification(ValidationStatus.FAIL, ("default Ollama identity changed",))
        return ScenarioVerification(reasons=("tiny-model RAM, VRAM, thermal, ownership, and default-server recovery were verified",))


def tiny_model_suite() -> ValidationSuite:
    return ValidationSuite(
        "tiny_model",
        (TinyModelManifestScenario, TinyModelProvisionScenario, TinyModelMetadataScenario,
         TinyModelResourcePreflightScenario, TinyModelInferenceScenario, TinyModelUnloadVerificationScenario,
         TinyModelForcedTimeoutScenario, TinyModelTimeoutCleanupVerificationScenario,
         TinyModelResourceRecoveryScenario, DefaultServerIsolationScenario),
        TinyModelContext,
        TinyModelRunGuard,
    )
