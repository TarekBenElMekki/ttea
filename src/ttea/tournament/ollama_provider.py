"""Closed-catalog Ollama provider using the validated benchmark substrate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable

from ttea.benchmarking import (
    BenchmarkOutcome, BenchmarkResult, BenchmarkSafetyPolicy, BenchmarkStateStore,
    BenchmarkSupervisor, OllamaBenchmarkWorker, OllamaClient, OllamaEndpoint,
    OllamaGenerationResult, OwnedOllamaServer, resolve_ollama_executable,
)
from ttea.benchmarking.ollama import OllamaCleanup, OllamaRuntimeError
from ttea.resources import ResourceDecision, ResourceManager, ResourceRequest
from ttea.validation.suites import listening_identity, loopback_port_open, wait_for_port_closed

from .execution import ProvisioningEvidence, SurvivalExecutionPolicy
from .manifest import TournamentManifest
from .models import TournamentCandidate, TournamentRound, TournamentTask
from .runner import (
    ProviderAvailability, ProviderMetadata, ProvisioningEligibility,
    TournamentExecutionRequest, TournamentProvider,
)


DEDICATED_ENDPOINT = OllamaEndpoint("http://127.0.0.1:11435")
DEFAULT_OLLAMA_PORT = 11434
DEDICATED_OLLAMA_PORT = 11435

# Survival admission is deliberately separate from catalog research-tier
# estimates. These values represent minimum safe headroom for attempting a
# CPU/GPU-offloaded cold load; actual RAM/VRAM use is measured empirically.
SURVIVAL_RAM_START_HEADROOM_BYTES = 512 * 1024 * 1024
SURVIVAL_GPU_HEADROOM_BYTES = 256 * 1024 * 1024


def survival_resource_request(candidate: TournamentCandidate) -> ResourceRequest:
    """Build the canonical, provider-layer survival admission request."""
    if not isinstance(candidate, TournamentCandidate):
        raise ValueError("candidate must be a TournamentCandidate")
    if candidate.variant.provider != "ollama":
        raise ValueError("survival request requires an Ollama candidate")
    return ResourceRequest(
        workload_type=f"tournament-survival:{candidate.candidate_id}",
        estimated_ram_bytes=SURVIVAL_RAM_START_HEADROOM_BYTES,
        estimated_disk_bytes=0,
        estimated_gpu_vram_bytes=SURVIVAL_GPU_HEADROOM_BYTES,
    )


class ProvisioningDeferred(RuntimeError):
    """A missing candidate cannot be acquired under the persisted policy."""


class ProvisioningRejected(RuntimeError):
    """Provisioning evidence failed closed-world identity verification."""


class ProvisioningResourceDelayed(RuntimeError):
    """Current resources cannot safely support acquisition yet."""


class ProvisioningRuntimeDelayed(RuntimeError):
    """Owned local runtime startup was transiently unavailable."""


class ProvisioningResourceDenied(RuntimeError):
    """Static resources cannot support acquisition."""


@dataclass(frozen=True)
class SurvivalExecutionResult:
    benchmark: BenchmarkResult
    metrics: dict[str, Any]
    cleanup_verified: bool
    resource_recovery_verified: bool
    default_server_identity_preserved: bool
    owned_ollama_pid: int | None = None
    owned_ollama_started_epoch: float | None = None


@dataclass
class _TelemetryTracker:
    gpu_index: int | None
    ram_lowest: int
    gpu_free_minimum: int | None
    gpu_temperature_maximum: float | None

    def observe(self, budget) -> None:
        self.ram_lowest = min(self.ram_lowest, budget.available_ram_bytes)
        gpu = next((item for item in budget.gpus if item.index == self.gpu_index), None)
        if gpu is None:
            return
        if gpu.free_vram_bytes is not None:
            self.gpu_free_minimum = (gpu.free_vram_bytes if self.gpu_free_minimum is None
                                     else min(self.gpu_free_minimum, gpu.free_vram_bytes))
        if gpu.temperature_c is not None:
            self.gpu_temperature_maximum = (gpu.temperature_c if self.gpu_temperature_maximum is None
                                             else max(self.gpu_temperature_maximum, gpu.temperature_c))


class OllamaTournamentProvider(TournamentProvider):
    """Provision and execute only candidates from one persisted manifest."""

    def __init__(self, manifest: TournamentManifest, resource_manager: ResourceManager,
                 execution_policy: SurvivalExecutionPolicy, *, executable: str | Path | None = None,
                 state_path: Path = Path("data/runtime/benchmark.json"), server_factory=OwnedOllamaServer,
                 default_identity: Callable[[int], Any] = listening_identity,
                 port_open: Callable[[int], bool] = loopback_port_open,
                 wait_port_closed: Callable[[int], bool] = wait_for_port_closed,
                 sleep: Callable[[float], None] = time.sleep, settle_seconds: float = 1.0) -> None:
        self.manifest = manifest
        self.resource_manager = resource_manager
        self.execution_policy = execution_policy
        self.executable = resolve_ollama_executable(executable)
        self.state_path = Path(state_path)
        self.server_factory = server_factory
        self.default_identity = default_identity
        self.port_open = port_open
        self.wait_port_closed = wait_port_closed
        self.sleep = sleep
        self.settle_seconds = settle_seconds
        self._allowed = {candidate.candidate_id: candidate.variant.model for candidate in manifest.candidates}
        if not self._allowed or any(candidate.variant.provider != "ollama" for candidate in manifest.candidates):
            raise ValueError("Ollama tournament manifest must contain only Ollama candidates")

    def _require_candidate(self, candidate: TournamentCandidate) -> None:
        if not isinstance(candidate, TournamentCandidate) or self._allowed.get(candidate.candidate_id) != candidate.variant.model:
            raise ProvisioningRejected("candidate is not the exact persisted manifest candidate")

    def availability(self, candidate: TournamentCandidate) -> ProviderAvailability:
        self._require_candidate(candidate)
        if self.executable is None:
            return ProviderAvailability(False, "Ollama executable is unavailable")
        return ProviderAvailability(True, f"Ollama executable resolved to {self.executable}")

    def metadata(self, candidate: TournamentCandidate) -> ProviderMetadata:
        self._require_candidate(candidate)
        return ProviderMetadata("ollama", candidate.variant.model, {
            "quantization": candidate.variant.quantization,
            "parameter_size": candidate.variant.parameter_size,
        })

    def provisioning_eligibility(self, candidate: TournamentCandidate) -> ProvisioningEligibility:
        availability = self.availability(candidate)
        return ProvisioningEligibility(availability.available, availability.reason)

    def resource_request(
        self,
        candidate: TournamentCandidate,
        tournament_round: TournamentRound,
        task: TournamentTask,
    ) -> ResourceRequest:
        """Admission headroom for starting measured survival execution.

        Published model package size remains catalog/disk metadata and is not
        treated as runtime RAM. The BenchmarkSupervisor observes real RAM,
        VRAM, and thermal pressure during model load and inference.
        """
        self._require_candidate(candidate)
        return survival_resource_request(candidate)


    def provisioning_resource_request(self, candidate: TournamentCandidate) -> ResourceRequest:
        self._require_candidate(candidate)
        return ResourceRequest(
            f"tournament-provision:{candidate.candidate_id}", 256 * 1024 * 1024,
            candidate.estimated_disk_bytes or 0,
        )

    def build_execution_request(self, candidate: TournamentCandidate, tournament_round: TournamentRound,
                                task: TournamentTask, repetition: int) -> TournamentExecutionRequest:
        self._require_candidate(candidate)
        policy = self.execution_policy
        if tournament_round.round_id != "survival_4k" or task.task_id not in tournament_round.task_ids:
            raise ValueError("only the persisted survival_4k round is executable")
        return TournamentExecutionRequest(
            candidate.candidate_id, tournament_round.round_id, task.task_id, repetition,
            "ollama", candidate.variant.model, task.prompt, policy.context_limit,
            policy.output_limit, policy.model_timeout_seconds, policy.temperature,
            policy.cpu_threads, policy.keep_alive,
        )

    @staticmethod
    def _inspect(client: OllamaClient, candidate: TournamentCandidate,
                 *, downloaded: bool, now: datetime) -> ProvisioningEvidence | None:
        exact = candidate.variant.model
        record = next((item for item in client.installed_model_records()
                       if (item.get("name") or item.get("model")) == exact), None)
        if record is None:
            return None
        digest, size = record.get("digest"), record.get("size")
        if not isinstance(digest, str) or not digest.strip():
            raise ProvisioningRejected("installed model record has no valid digest")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProvisioningRejected("installed model record has no valid size")
        shown = client.show_model(exact)
        details = shown.get("details") if isinstance(shown.get("details"), dict) else {}
        listed_details = record.get("details") if isinstance(record.get("details"), dict) else {}
        parameter = details.get("parameter_size") or listed_details.get("parameter_size")
        quantization = details.get("quantization_level") or listed_details.get("quantization_level")
        if quantization and candidate.variant.quantization and quantization.lower() != candidate.variant.quantization.lower():
            raise ProvisioningRejected("installed model quantization does not match manifest")
        capabilities = shown.get("capabilities", ())
        if not isinstance(capabilities, list) or any(not isinstance(item, str) or not item.strip() for item in capabilities):
            capabilities = ()
        return ProvisioningEvidence(
            candidate.candidate_id, exact, "ollama", digest, size,
            parameter if isinstance(parameter, str) and parameter.strip() else None,
            quantization if isinstance(quantization, str) and quantization.strip() else None,
            tuple(capabilities), now, not downloaded, downloaded,
        )

    def provision_with_client(self, client: OllamaClient, candidate: TournamentCandidate,
                              prior: ProvisioningEvidence | None = None,
                              acquisition_preflight: Callable[[ResourceRequest], Any] | None = None) -> ProvisioningEvidence:
        """Provision one exact candidate through an already-owned endpoint."""
        self._require_candidate(candidate)
        now = datetime.now(timezone.utc)
        existing = self._inspect(client, candidate, downloaded=False, now=now)
        if existing is not None:
            if prior is not None and (prior.model != existing.model or prior.digest != existing.digest):
                raise ProvisioningRejected("installed model digest or identity differs from persisted evidence")
            return prior or existing
        if prior is not None:
            raise ProvisioningRejected("previously verified model is no longer installed")
        if acquisition_preflight is not None:
            assessment = acquisition_preflight(self.provisioning_resource_request(candidate))
            if assessment.decision is ResourceDecision.DELAY:
                raise ProvisioningResourceDelayed("; ".join(assessment.reasons))
            if assessment.decision is ResourceDecision.DENY:
                raise ProvisioningResourceDenied("; ".join(assessment.reasons))
        if not self.execution_policy.allow_model_download:
            raise ProvisioningDeferred("model is missing and model download was not authorized")
        client.pull(candidate.variant.model, self.execution_policy.pull_timeout_seconds)
        verified = self._inspect(client, candidate, downloaded=True, now=datetime.now(timezone.utc))
        if verified is None:
            raise ProvisioningRejected("exact model remains absent after authorized pull")
        return verified

    def _server(self, policy: BenchmarkSafetyPolicy, state_store: BenchmarkStateStore,
                identity_observer: Callable[[Any], None] | None = None) -> OwnedOllamaServer:
        def record_identity(identity) -> None:
            state_store.record_server(identity.pid, identity.started_epoch)
            if identity_observer is not None:
                identity_observer(identity)

        return self.server_factory(
            DEDICATED_ENDPOINT, policy, executable=self.executable,
            identity_callback=record_identity,
        )

    def provision(self, candidate: TournamentCandidate,
                  prior: ProvisioningEvidence | None = None,
                  acquisition_preflight: Callable[[ResourceRequest], Any] | None = None) -> ProvisioningEvidence:
        self._require_candidate(candidate)
        if self.port_open(DEDICATED_OLLAMA_PORT):
            raise ProvisioningDeferred("dedicated Ollama port 11435 is occupied")
        default_before = self.default_identity(DEFAULT_OLLAMA_PORT)
        policy = self.benchmark_policy()
        state_store = BenchmarkStateStore(self.state_path)
        state_store.acquire()
        server = self._server(policy, state_store)
        try:
            started = server.start()
            if not started.success:
                reason = "; ".join(started.reasons)
                if "startup timed out" in reason.lower():
                    raise ProvisioningRuntimeDelayed(reason)
                raise OllamaRuntimeError(reason)
            evidence = self.provision_with_client(server.client(), candidate, prior, acquisition_preflight)
            return evidence
        finally:
            try:
                server.shutdown(policy.cooldown_timeout_seconds)
            finally:
                if server.shutdown_confirmed:
                    state_store.release()
            if not self.wait_port_closed(DEDICATED_OLLAMA_PORT):
                raise OllamaRuntimeError("dedicated port remained open after provisioning")
            if self.default_identity(DEFAULT_OLLAMA_PORT) != default_before:
                raise ProvisioningRejected("default Ollama listener identity changed during provisioning")

    def benchmark_policy(self) -> BenchmarkSafetyPolicy:
        policy = self.execution_policy
        return BenchmarkSafetyPolicy(
            telemetry_interval_seconds=1.0, consecutive_resource_violation_samples=3,
            request_timeout_seconds=policy.request_timeout_seconds,
            model_timeout_seconds=policy.model_timeout_seconds, cooldown_timeout_seconds=10.0,
            gpu_warning_temperature_c=70.0, gpu_abort_temperature_c=80.0,
            max_context_tokens=policy.context_limit, max_output_tokens=policy.output_limit,
            cpu_thread_limit=policy.cpu_threads,
        )

    def execute_prompt(self, candidate: TournamentCandidate,
                         request: TournamentExecutionRequest,
                         resource_request: ResourceRequest) -> SurvivalExecutionResult:
        """Run one exact manifest candidate through the supervised owned-runtime path."""
        self._require_candidate(candidate)
        if (request.candidate_id, request.model) != (candidate.candidate_id, candidate.variant.model):
            raise ProvisioningRejected("execution request does not match persisted manifest")
        default_before = self.default_identity(DEFAULT_OLLAMA_PORT)
        policy = self.benchmark_policy()
        state_store = BenchmarkStateStore(self.state_path)
        state_store.acquire()
        owned_identity = None

        def observe_identity(identity) -> None:
            nonlocal owned_identity
            owned_identity = identity

        server = self._server(policy, state_store, observe_identity)
        cleanup = None
        handed_off = False
        before = after = None
        tracker = None
        benchmark = None
        model_cleanup = None
        try:
            startup = server.start()
            if not startup.success:
                reason = "; ".join(startup.reasons)
                if "startup timed out" in reason.lower():
                    raise ProvisioningRuntimeDelayed(reason)
                raise OllamaRuntimeError(reason)
            client = server.client()
            if candidate.variant.model not in client.installed_models():
                raise ProvisioningRejected("verified candidate is not installed in the owned runtime")
            if candidate.variant.model in client.running_models():
                client.unload(candidate.variant.model)
            if candidate.variant.model in client.running_models():
                raise OllamaRuntimeError("candidate was resident before cold survival workload")
            before = self.resource_manager.current_budget()
            selected_gpu = next((gpu for gpu in before.gpus if gpu.index == resource_request.gpu_index), None)
            tracker = _TelemetryTracker(
                resource_request.gpu_index, before.available_ram_bytes,
                None if selected_gpu is None else selected_gpu.free_vram_bytes,
                None if selected_gpu is None else selected_gpu.temperature_c,
            )
            model_cleanup = OllamaCleanup(client, server, candidate.variant.model, model_cleanup_required=False)
            cleanup = model_cleanup.bounded()
            supervisor = BenchmarkSupervisor(
                self.resource_manager, policy, cleanup, self.state_path,
                state_store=state_store, state_already_acquired=True,
                worker_started_callback=model_cleanup.mark_model_cleanup_required,
                telemetry_observer=tracker.observe,
            )
            handed_off = True
            worker = OllamaBenchmarkWorker(
                client, request.model, request.prompt, num_ctx=request.context_limit,
                num_predict=request.output_limit, num_thread=request.cpu_threads,
                temperature=request.temperature, keep_alive=request.keep_alive,
            )
            benchmark = supervisor.run(worker, resource_request)
            self.sleep(self.settle_seconds)
            after = self.resource_manager.current_budget()
        finally:
            if not handed_off:
                try:
                    if cleanup is not None:
                        cleanup.run(policy.cooldown_timeout_seconds)
                    else:
                        server.shutdown(policy.cooldown_timeout_seconds)
                finally:
                    if server.shutdown_confirmed:
                        state_store.release()

        if benchmark is None or before is None or after is None or tracker is None or model_cleanup is None:
            raise OllamaRuntimeError("survival execution did not produce supervised evidence")
        selected_before = next((gpu for gpu in before.gpus if gpu.index == resource_request.gpu_index), None)
        selected_after = next((gpu for gpu in after.gpus if gpu.index == resource_request.gpu_index), None)
        gpu_before = None if selected_before is None else selected_before.free_vram_bytes
        gpu_after = None if selected_after is None else selected_after.free_vram_bytes
        ram_delta = after.available_ram_bytes - before.available_ram_bytes
        gpu_delta = None if gpu_before is None or gpu_after is None else gpu_after - gpu_before
        ram_tolerance = max(256 * 1024 * 1024, int(before.available_ram_bytes * 0.05))
        gpu_tolerance = None if gpu_before is None else max(128 * 1024 * 1024, int(gpu_before * 0.05))
        recovered = ram_delta >= -ram_tolerance and (gpu_delta is None or gpu_delta >= -(gpu_tolerance or 0))
        port_closed = not self.port_open(DEDICATED_OLLAMA_PORT)
        runtime_state_absent = not self.state_path.exists()
        default_preserved = self.default_identity(DEFAULT_OLLAMA_PORT) == default_before
        cleanup_verified = model_cleanup.model_unload_confirmed and server.shutdown_confirmed and port_closed and runtime_state_absent
        metrics: dict[str, Any] = {
            "ram_available_before_bytes": before.available_ram_bytes,
            "ram_minimum_available_bytes": tracker.ram_lowest,
            "ram_available_after_bytes": after.available_ram_bytes,
            "ram_transient_delta_bytes": before.available_ram_bytes - tracker.ram_lowest,
            "ram_final_delta_bytes": ram_delta,
            "gpu_index": resource_request.gpu_index,
            "gpu_free_vram_before_bytes": gpu_before,
            "gpu_minimum_free_vram_bytes": tracker.gpu_free_minimum,
            "gpu_free_vram_after_bytes": gpu_after,
            "gpu_transient_vram_delta_bytes": (None if gpu_before is None or tracker.gpu_free_minimum is None
                                                else gpu_before - tracker.gpu_free_minimum),
            "gpu_final_vram_delta_bytes": gpu_delta,
            "max_gpu_temperature_c": tracker.gpu_temperature_maximum,
            "wall_clock_duration_seconds": (benchmark.finished_at - benchmark.started_at).total_seconds(),
            "worker_started": benchmark.worker_started,
            "timeout_kind": benchmark.timeout_kind,
            "model_unload_confirmed": model_cleanup.model_unload_confirmed,
            "server_shutdown_confirmed": server.shutdown_confirmed,
            "dedicated_port_closed": port_closed,
            "runtime_state_absent": runtime_state_absent,
        }
        generated = benchmark.worker_result
        if isinstance(generated, OllamaGenerationResult):
            decode_rate = None
            if generated.eval_count is not None and generated.eval_duration:
                decode_rate = generated.eval_count / (generated.eval_duration / 1_000_000_000)
            metrics.update({
                "response_is_string": isinstance(generated.response, str),
                "cold_load_latency_seconds": None if generated.load_duration is None else generated.load_duration / 1_000_000_000,
                "total_ollama_duration_seconds": None if generated.total_duration is None else generated.total_duration / 1_000_000_000,
                "load_duration": generated.load_duration,
                "prompt_eval_duration": generated.prompt_eval_duration,
                "eval_duration": generated.eval_duration,
                "eval_count": generated.eval_count,
                "prompt_eval_count": generated.prompt_eval_count,
                "decode_tokens_per_second": decode_rate,
            })
        return SurvivalExecutionResult(
            benchmark,
            metrics,
            cleanup_verified,
            recovered,
            default_preserved,
            None if owned_identity is None else owned_identity.pid,
            None if owned_identity is None else owned_identity.started_epoch,
        )

    def execute_survival(self, candidate: TournamentCandidate,
                         request: TournamentExecutionRequest,
                         resource_request: ResourceRequest) -> SurvivalExecutionResult:
        """Run one candidate only through BenchmarkSupervisor and its owned worker."""
        return self.execute_prompt(candidate, request, resource_request)
