from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ttea.benchmarking import BenchmarkOutcome, BenchmarkResult, BenchmarkSafetyPolicy
from ttea.benchmarking.ollama import OllamaGenerationResult, OllamaRuntimeError, OwnedProcessIdentity
from ttea.resources import ResourceDecision, ResourceRequest
from ttea.telemetry.models import (CpuMetrics, DiskMetrics, GpuMetrics, GpuProfile, HardwareProfile,
                                   HardwareSnapshot, MemoryMetrics)
from ttea.validation import (ValidationRunOptions, ValidationRunResult, ValidationScenarioResult,
                             ValidationStatus, freeze_evidence)
from ttea.validation.reporting import result_to_dict, text_report
from ttea.validation.model_assets import (ModelProvisionBlocked, TINY_MODEL_MANIFEST,
                                          TINY_MODEL_NAME, TinyModelMetadata,
                                          TinyModelProvisioner, validation_manifest)
from ttea.validation.suites import build_registry
from ttea.validation.tiny_model import (TinyModelContext, TinyModelForcedTimeoutScenario,
                                        TinyModelInferenceScenario, TinyModelProvisionScenario,
                                        TinyModelResourceRecoveryScenario,
                                        TinyModelTimeoutCleanupVerificationScenario,
                                        TinyModelUnloadVerificationScenario, tiny_policy)


def assessment(decision=ResourceDecision.ALLOW, reasons=("safe",)):
    return Mock(decision=decision, reasons=reasons)


def generation_result(outcome=BenchmarkOutcome.COMPLETED, *, worker_started=None, timeout_kind=None):
    generated = OllamaGenerationResult(TINY_MODEL_NAME, "hello", 10, 2, 3, 4, 5, 1_000_000_000, "stop")
    now = datetime.now(timezone.utc)
    if worker_started is None:
        worker_started = outcome is not BenchmarkOutcome.RESOURCE_PRESSURE
    return BenchmarkResult(outcome, ("result",), now, now,
                           worker_result=generated if outcome is BenchmarkOutcome.COMPLETED else None,
                           worker_started=worker_started, timeout_kind=timeout_kind)


class FakeAssetClient:
    def __init__(self, records):
        self.records = list(records)
        self.pull_calls = []
        self.show_payload = {"details": {"parameter_size": "135M", "quantization_level": "Q4_0"},
                             "capabilities": ["completion"]}
    def installed_model_records(self):
        value = self.records.pop(0) if len(self.records) > 1 else self.records[0]
        return tuple(value)
    def show_model(self, model):
        return self.show_payload
    def pull(self, model, timeout):
        self.pull_calls.append((model, timeout))
        return {"status": "success"}


class FakeTelemetry:
    def __init__(self, gpu=True):
        profile_gpu = (GpuProfile(0, "NVIDIA", 2 * 1024**3, "nvidia-smi"),) if gpu else ()
        metric_gpu = (GpuMetrics(0, "NVIDIA", 1, 2 * 1024**3, 1, 2 * 1024**3 - 1, 45, 5),) if gpu else ()
        self.profile = HardwareProfile("host", "os", "r", "v", "x64", "cpu", 4, 8,
                                       8 * 1024**3, 100 * 1024**3, profile_gpu)
        self.observation = HardwareSnapshot(datetime.now(timezone.utc), CpuMetrics(1, 1, None),
                                            MemoryMetrics(8 * 1024**3, 1, 7 * 1024**3, 10),
                                            DiskMetrics(100 * 1024**3, 1, 99 * 1024**3, 1, None, None),
                                            metric_gpu, 10)
    def get_profile(self): return self.profile
    def snapshot(self): return self.observation


class TinyModelTests(unittest.TestCase):
    def test_exact_manifest_allowlist_and_no_arbitrary_model(self):
        self.assertIs(validation_manifest(TINY_MODEL_NAME), TINY_MODEL_MANIFEST)
        self.assertEqual(TINY_MODEL_MANIFEST.ram_estimate_bytes, 512 * 1024 * 1024)
        self.assertEqual(TINY_MODEL_MANIFEST.gpu_vram_estimate_bytes, 256 * 1024 * 1024)
        with self.assertRaises(ValueError):
            validation_manifest("arbitrary:latest")

    def test_missing_without_authorization_blocks_and_existing_never_pulls(self):
        provisioner = TinyModelProvisioner()
        with self.assertRaises(ModelProvisionBlocked):
            provisioner.ensure(FakeAssetClient([()]), False)
        record = {"name": TINY_MODEL_NAME, "digest": "sha256:abc", "size": 123,
                  "details": {"parameter_size": "135M", "quantization_level": "Q4_0"}}
        client = FakeAssetClient([(record,)])
        metadata, downloaded = provisioner.ensure(client, False)
        self.assertFalse(downloaded)
        self.assertEqual((metadata.digest, metadata.size_bytes), ("sha256:abc", 123))
        self.assertEqual(metadata.capabilities, ("completion",))
        self.assertEqual(client.pull_calls, [])

    def test_resume_after_download_rediscovers_model_without_redownloading(self):
        record = {"name": TINY_MODEL_NAME, "digest": "sha256:resume", "size": 789, "details": {}}
        resumed_client = FakeAssetClient([(record,)])
        metadata, downloaded = TinyModelProvisioner().ensure(resumed_client, True)
        self.assertFalse(downloaded)
        self.assertEqual(metadata.digest, "sha256:resume")
        self.assertEqual(resumed_client.pull_calls, [])

    def test_authorized_provision_uses_exact_model_and_bounded_timeout(self):
        record = {"name": TINY_MODEL_NAME, "digest": "sha256:def", "size": 456, "details": {}}
        client = FakeAssetClient([(), (record,)])
        metadata, downloaded = TinyModelProvisioner().ensure(client, True)
        self.assertTrue(downloaded)
        self.assertEqual(client.pull_calls, [(TINY_MODEL_NAME, 20 * 60)])
        self.assertEqual(metadata.digest, "sha256:def")

    def test_malformed_installed_metadata_is_rejected(self):
        record = {"name": TINY_MODEL_NAME, "size": 1}
        with self.assertRaises(OllamaRuntimeError):
            TinyModelProvisioner().inspect(FakeAssetClient([(record,)]))

    def test_gpu_request_is_conservative_and_cpu_does_not_invent_gpu(self):
        gpu_context = TinyModelContext(telemetry=FakeTelemetry(True), resource_manager=Mock())
        request = gpu_context.build_inference_request()
        self.assertEqual(request.estimated_ram_bytes, 512 * 1024 * 1024)
        self.assertEqual(request.estimated_gpu_vram_bytes, 256 * 1024 * 1024)
        self.assertEqual(request.gpu_index, 0)
        cpu_context = TinyModelContext(telemetry=FakeTelemetry(False), resource_manager=Mock())
        request = cpu_context.build_inference_request()
        self.assertEqual(request.estimated_gpu_vram_bytes, 0)
        self.assertIsNone(request.gpu_index)

    def test_tiny_policy_enforces_generation_bounds(self):
        policy = tiny_policy()
        self.assertEqual((policy.max_context_tokens, policy.max_output_tokens, policy.cpu_thread_limit), (512, 16, 2))
        self.assertGreater(policy.request_timeout_seconds, tiny_policy(0.1).model_timeout_seconds)

    def test_provision_scenario_blocks_without_download_and_checks_disk_before_pull(self):
        server = Mock()
        client = Mock()
        server.client.return_value = client
        context = Mock(pre_model_passed=True, allow_model_download=False, manifest=TINY_MODEL_MANIFEST,
                       provisioner=Mock(), resource_manager=Mock())
        context.resource_manager.assess.return_value = assessment()
        context.acquire_server.return_value = server
        context.provisioner.inspect.return_value = None
        result = TinyModelProvisionScenario().run(context)
        self.assertIs(result.status, ValidationStatus.BLOCKED)
        context.provisioner.ensure.assert_not_called()
        context.release_server_only.assert_called_once()

        metadata = TinyModelMetadata(TINY_MODEL_NAME, "sha256:x", 1)
        context = Mock(pre_model_passed=True, allow_model_download=True, manifest=TINY_MODEL_MANIFEST,
                       provisioner=Mock(), resource_manager=Mock())
        context.acquire_server.return_value = server
        context.provisioner.inspect.return_value = None
        context.provisioner.ensure.return_value = (metadata, True)
        context.resource_manager.assess.return_value = assessment()
        result = TinyModelProvisionScenario().run(context)
        self.assertIs(result.status, ValidationStatus.PASS)
        disk_request = context.resource_manager.assess.call_args.args[0]
        self.assertEqual(disk_request.estimated_disk_bytes, TINY_MODEL_MANIFEST.expected_disk_bytes)

    def test_owned_provision_server_is_dedicated(self):
        with tempfile.TemporaryDirectory() as directory:
            context = TinyModelContext(telemetry=FakeTelemetry(False), resource_manager=Mock(),
                                       state_path=Path(directory) / "state.json", state_store=Mock())
            fake_server = Mock()
            identity = OwnedProcessIdentity(10, 20)
            fake_server.start.return_value = Mock(success=True, identity=identity)
            with patch("ttea.validation.tiny_model.OwnedOllamaServer", return_value=fake_server) as server_type, \
                 patch("ttea.validation.tiny_model.loopback_port_open", return_value=False), \
                 patch("ttea.validation.tiny_model.listening_identity", return_value=identity):
                context.acquire_server(tiny_policy())
            self.assertEqual(server_type.call_args.args[0].base_url, "http://127.0.0.1:11435")
            context.state_store.acquire.assert_called_once()

    def test_inference_delay_and_completed_metrics_contract(self):
        request = ResourceRequest("tiny", 1, 0)
        context = Mock(inference_request=request, resource_manager=Mock())
        context.resource_manager.assess.return_value = assessment(ResourceDecision.DELAY, ("low RAM",))
        result = TinyModelInferenceScenario().run(context)
        self.assertIs(result.status, ValidationStatus.BLOCKED)
        context.run_generation.assert_not_called()

        evidence = {"response_is_string": True, "eval_count": 5, "eval_duration": 1_000_000_000,
                    "tokens_per_second": 5.0}
        context = Mock(inference_request=request, resource_manager=Mock())
        context.resource_manager.assess.return_value = assessment()
        context.run_generation.return_value = (generation_result(), evidence)
        result = TinyModelInferenceScenario().run(context)
        self.assertIs(result.status, ValidationStatus.PASS)
        self.assertEqual(dict(result.evidence)["tokens_per_second"], 5.0)

    def test_forced_timeout_and_cleanup_verification(self):
        request = ResourceRequest("tiny", 1, 0)
        context = Mock(normal_result=generation_result(), inference_request=request, resource_manager=Mock())
        context.resource_manager.assess.return_value = assessment()
        cleanup = {"model_unload_confirmed": True, "server_shutdown_confirmed": True,
                   "port_closed": True, "runtime_state_absent": True}
        context.run_generation.return_value = (generation_result(BenchmarkOutcome.TIMEOUT,
                                                                  worker_started=True, timeout_kind="model"), cleanup)
        timed = TinyModelForcedTimeoutScenario().run(context)
        self.assertIs(timed.status, ValidationStatus.PASS)
        self.assertEqual(context.run_generation.call_args.args[0].model_timeout_seconds, 0.25)
        context.timeout_evidence = cleanup
        verified = TinyModelTimeoutCleanupVerificationScenario().run(context)
        self.assertIs(verified.status, ValidationStatus.PASS)
        context.normal_evidence = cleanup
        self.assertIs(TinyModelUnloadVerificationScenario().run(context).status, ValidationStatus.PASS)

    def test_forced_timeout_cannot_pass_on_prestart_request_timeout(self):
        request = ResourceRequest("tiny", 1, 0)
        context = Mock(normal_result=generation_result(), inference_request=request, resource_manager=Mock())
        context.resource_manager.assess.return_value = assessment()
        cleanup = {"model_unload_confirmed": True, "server_shutdown_confirmed": True,
                   "port_closed": True, "runtime_state_absent": True,
                   "worker_started": False, "timeout_kind": "request"}
        context.run_generation.return_value = (
            generation_result(BenchmarkOutcome.TIMEOUT, worker_started=False, timeout_kind="request"), cleanup)
        result = TinyModelForcedTimeoutScenario().run(context)
        self.assertIs(result.status, ValidationStatus.FAIL)
        self.assertIn("post-start model deadline", " ".join(result.reasons))

    def test_registry_includes_tiny_model_after_pre_model(self):
        registry = build_registry()
        self.assertEqual(registry.names(), ("pre_model", "tiny_model"))
        suite = registry.get("tiny_model")
        self.assertEqual(suite.scenario_factories[0]().scenario_id, "tiny_model_manifest")
        self.assertEqual(suite.scenario_factories[-1]().scenario_id, "default_server_isolation")

    def test_resource_recovery_emits_machine_and_human_safety_summary(self):
        metadata = TinyModelMetadata(TINY_MODEL_NAME, "sha256:summary", 321, "135M", "Q4_0", ("completion",))
        operation = {"resources_recovered": True, "thermal_safe": True, "eval_count": 5,
                     "tokens_per_second": 5.0, "ram_delta_bytes": -10, "gpu_delta_bytes": 0,
                     "gpu_max_temperature_c": 50.0, "model_unload_confirmed": True,
                     "server_shutdown_confirmed": True, "port_closed": True, "runtime_state_absent": True}
        context = Mock(metadata=metadata, normal_result=generation_result(), normal_evidence=operation,
                       timeout_result=generation_result(BenchmarkOutcome.TIMEOUT,
                                                        worker_started=True, timeout_kind="model"), timeout_evidence=operation,
                       default_identity=None)
        result = TinyModelResourceRecoveryScenario().run(context)
        self.assertIs(result.status, ValidationStatus.PASS)
        now = datetime.now(timezone.utc)
        run = ValidationRunResult("tiny-run", "tiny_model", now, now, ValidationStatus.PASS,
                                  (result,), freeze_evidence({}), True)
        payload = result_to_dict(run)
        self.assertEqual(payload["safety_summary"]["digest"], "sha256:summary")
        self.assertTrue(payload["safety_summary"]["timeout_cleanup_verified"])
        self.assertIn("Tiny-model safety summary", text_report(run))


if __name__ == "__main__":
    unittest.main()
