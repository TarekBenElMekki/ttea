from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ttea.benchmarking import BenchmarkOutcome, BenchmarkResult, OllamaBenchmarkWorker, OllamaClient, OwnedOllamaServer
from ttea.benchmarking.ollama import OllamaRuntimeError
from ttea.resources import ResourceDecision, ResourceRequest
from ttea.tournament import (
    CandidateSurvivalRecord, CandidateVariant, OllamaTournamentProvider,
    ProvisioningDeferred, ProvisioningEvidence, ProvisioningRejected,
    ProvisioningRuntimeDelayed, ProvisioningState,
    SurvivalCampaign, SurvivalCampaignState,
    SurvivalExecutionPolicy, SurvivalExecutionResult, SurvivalOutcome,
    SurvivalStore, SurvivalTournamentEngine, TournamentCandidate,
    TournamentManifest, TournamentSupervisorManager, build_survival_manifest,
    survival_campaign_from_dict, survival_campaign_to_dict,
    survival_report_payload,
)
from ttea.tournament.__main__ import main as tournament_main
from ttea.tournament.ollama_provider import DEDICATED_ENDPOINT
from ttea.tournament.watch import TournamentWatchRenderer, TournamentWatcher, watch


NOW = datetime.now(timezone.utc) - timedelta(minutes=1)


class Clock:
    def __init__(self):
        self.value = NOW

    def now(self):
        self.value += timedelta(milliseconds=1)
        return self.value

    def sleep(self, seconds):
        self.value += timedelta(seconds=seconds)


class FakeResourceManager:
    def __init__(self, decisions=(), selected_gpu_index=None):
        self.decisions = list(decisions)
        self.selected_gpu_index = selected_gpu_index
        self.calls = []

    def assess(self, request):
        self.calls.append(request)
        decision = self.decisions.pop(0) if self.decisions else ResourceDecision.ALLOW
        return SimpleNamespace(
            decision=decision,
            reasons=(f"resource {decision.value.lower()}",),
            selected_gpu_index=self.selected_gpu_index,
        )


def evidence(candidate, *, digest="sha256:good", downloaded=False):
    return ProvisioningEvidence(
        candidate.candidate_id, candidate.variant.model, "ollama", digest, 1234,
        candidate.variant.parameter_size, candidate.variant.quantization, ("completion",),
        NOW, not downloaded, downloaded,
    )


def benchmark(outcome=BenchmarkOutcome.COMPLETED, *, worker_started=True, timeout_kind=None):
    return BenchmarkResult(outcome, (outcome.value.lower(),), NOW, NOW + timedelta(seconds=1),
                           321, None, worker_started, timeout_kind)


class FakeProvider:
    def __init__(self, manifest, resource_manager, policy, executions=()):
        self.manifest, self.resource_manager, self.execution_policy = manifest, resource_manager, policy
        self.executions = list(executions)
        self.provision_calls = []
        self.execute_calls = []
        self.execute_requests = []
        self.concurrent = self.maximum_concurrent = 0

    def provision(self, candidate, prior=None, acquisition_preflight=None):
        self.provision_calls.append((candidate.candidate_id, prior))
        return prior or evidence(candidate)

    def resource_request(self, candidate, tournament_round, task):
        return ResourceRequest(
            f"survival:{candidate.candidate_id}",
            1,
            0,
            estimated_gpu_vram_bytes=1,
        )

    def build_execution_request(self, candidate, tournament_round, task, repetition):
        from ttea.tournament import TournamentExecutionRequest
        return TournamentExecutionRequest(candidate.candidate_id, tournament_round.round_id, task.task_id,
                                          repetition, "ollama", candidate.variant.model, task.prompt,
                                          4096, 32, 180, 0, 4, 0)

    def execute_survival(self, candidate, request, resource_request):
        self.concurrent += 1
        self.maximum_concurrent = max(self.maximum_concurrent, self.concurrent)
        self.execute_calls.append(candidate.candidate_id)
        self.execute_requests.append(resource_request)
        try:
            if self.executions:
                return self.executions.pop(0)
            return SurvivalExecutionResult(benchmark(), {"cold_load_latency_seconds": 0.5,
                                                          "decode_tokens_per_second": 10.0}, True, True, True)
        finally:
            self.concurrent -= 1


class FakeClient:
    def __init__(self, model, *, installed=True, digest="sha256:good", quantization="Q4_K_M",
                 pull_error=None):
        self.model, self.installed, self.digest = model, installed, digest
        self.quantization, self.pull_error = quantization, pull_error
        self.pulls = []

    def installed_model_records(self):
        if not self.installed:
            return ()
        return ({"name": self.model, "digest": self.digest, "size": 1234,
                 "details": {"parameter_size": "1B", "quantization_level": self.quantization}},)

    def show_model(self, model):
        return {"details": {"parameter_size": "1B", "quantization_level": self.quantization},
                "capabilities": ["completion"]}

    def pull(self, model, timeout):
        self.pulls.append((model, timeout))
        if self.pull_error:
            raise self.pull_error
        self.installed = True
        return {"status": "success"}


class TournamentSurvivalTests(unittest.TestCase):
    def manifest(self, count=1):
        source = build_survival_manifest()
        return TournamentManifest(source.candidates[:count], source.rounds, source.tasks)

    def create(self, directory, *, count=1, allow=False, policy=None):
        manifest = self.manifest(count)
        policy = policy or SurvivalExecutionPolicy(allow_model_download=allow, resource_retry_seconds=0.5)
        campaign = SurvivalCampaign(
            "survival-test", manifest.digest, SurvivalCampaignState.CREATED, NOW, NOW,
            NOW + timedelta(hours=6),
            {item.candidate_id: CandidateSurvivalRecord(item.candidate_id, item.variant.model)
             for item in manifest.candidates},
        )
        store = SurvivalStore(directory)
        store.create(campaign, manifest, policy)
        return store, campaign, manifest, policy

    def run_engine(self, directory, *, count=1, decisions=(), executions=()):
        store, campaign, manifest, policy = self.create(directory, count=count)
        resources = FakeResourceManager(decisions)
        provider = FakeProvider(manifest, resources, policy, executions)
        clock = Clock()
        result = SurvivalTournamentEngine(campaign.tournament_id, store, provider,
                                          now=clock.now, sleep=clock.sleep).run()
        return result, store, provider, resources

    def test_survival_manifest_is_closed_ordered_and_exact(self):
        manifest = build_survival_manifest()
        self.assertEqual(len(manifest.candidates), 13)
        self.assertEqual(manifest.rounds[0].round_id, "survival_4k")
        self.assertEqual(manifest.tasks[0].prompt, "Reply exactly with: TTEA_SURVIVAL_OK")
        self.assertFalse(any(candidate.validation_only for candidate in manifest.candidates))

    def test_execution_policy_is_bounded_and_download_default_is_false(self):
        policy = SurvivalExecutionPolicy()
        self.assertFalse(policy.allow_model_download)
        self.assertEqual((policy.context_limit, policy.output_limit, policy.temperature,
                          policy.keep_alive, policy.max_concurrency), (4096, 32, 0, 0, 1))
        self.assertEqual((policy.request_timeout_seconds, policy.model_timeout_seconds), (300, 180))

    def test_policy_refuses_unsupported_phase_or_unbounded_threads(self):
        with self.assertRaises(ValueError):
            SurvivalExecutionPolicy(through="capability")
        with self.assertRaises(ValueError):
            SurvivalExecutionPolicy(cpu_threads=99)

    def provider(self, manifest, resources=None, allow=False):
        return OllamaTournamentProvider(manifest, resources or FakeResourceManager(),
                                        SurvivalExecutionPolicy(allow_model_download=allow),
                                        executable=Path(__file__), port_open=lambda port: False,
                                        default_identity=lambda port: None,
                                        wait_port_closed=lambda port: True, settle_seconds=0)

    def test_installed_model_is_reused_without_pull(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        client = FakeClient(candidate.variant.model, quantization=candidate.variant.quantization)
        found = self.provider(manifest).provision_with_client(client, candidate)
        self.assertTrue(found.already_installed)
        self.assertFalse(found.download_attempted)
        self.assertEqual(client.pulls, [])

    def test_missing_model_without_authorization_is_blocked_after_preflight(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        client = FakeClient(candidate.variant.model, installed=False,
                            quantization=candidate.variant.quantization)
        preflight = Mock(return_value=SimpleNamespace(decision=ResourceDecision.ALLOW, reasons=("ok",)))
        with self.assertRaises(ProvisioningDeferred):
            self.provider(manifest).provision_with_client(client, candidate,
                                                          acquisition_preflight=preflight)
        preflight.assert_called_once()
        self.assertEqual(client.pulls, [])

    def test_authorized_pull_uses_only_exact_manifest_tag_and_timeout(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        client = FakeClient(candidate.variant.model, installed=False,
                            quantization=candidate.variant.quantization)
        found = self.provider(manifest, allow=True).provision_with_client(
            client, candidate,
            acquisition_preflight=lambda request: SimpleNamespace(decision=ResourceDecision.ALLOW, reasons=("ok",)),
        )
        self.assertEqual(client.pulls, [(candidate.variant.model, 1800)])
        self.assertTrue(found.download_attempted)

    def test_arbitrary_candidate_is_rejected(self):
        manifest = self.manifest()
        arbitrary = TournamentCandidate("arbitrary", "x", CandidateVariant("ollama", "evil:latest", "q4", "1B"),
                                        1, 1, 0, 4096, "not persisted")
        with self.assertRaises(ProvisioningRejected):
            self.provider(manifest).availability(arbitrary)

    def test_persisted_digest_mismatch_is_rejected(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        client = FakeClient(candidate.variant.model, digest="sha256:different",
                            quantization=candidate.variant.quantization)
        with self.assertRaises(ProvisioningRejected):
            self.provider(manifest).provision_with_client(client, candidate, evidence(candidate))

    def test_verified_restart_does_not_repull(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        prior = evidence(candidate)
        client = FakeClient(candidate.variant.model, quantization=candidate.variant.quantization)
        found = self.provider(manifest, allow=True).provision_with_client(client, candidate, prior)
        self.assertIs(found, prior)
        self.assertEqual(client.pulls, [])

    def test_pull_timeout_propagates_without_identity_fabrication(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        client = FakeClient(candidate.variant.model, installed=False,
                            quantization=candidate.variant.quantization,
                            pull_error=OllamaRuntimeError("pull timeout"))
        with self.assertRaisesRegex(OllamaRuntimeError, "timeout"):
            self.provider(manifest, allow=True).provision_with_client(
                client, candidate,
                acquisition_preflight=lambda request: SimpleNamespace(decision=ResourceDecision.ALLOW, reasons=("ok",)),
            )

    def test_provider_uses_dedicated_endpoint_and_never_default_for_pull(self):
        self.assertEqual(DEDICATED_ENDPOINT.base_url, "http://127.0.0.1:11435")
        self.assertNotIn("11434", DEDICATED_ENDPOINT.base_url)

    def test_execution_request_and_worker_forward_survival_controls(self):
        manifest = self.manifest()
        candidate, round_value, task = manifest.candidates[0], manifest.rounds[0], manifest.tasks[0]
        request = self.provider(manifest).build_execution_request(candidate, round_value, task, 0)
        self.assertEqual((request.context_limit, request.output_limit, request.temperature,
                          request.cpu_threads, request.keep_alive), (4096, 32, 0, 4, 0))
        client = Mock()
        OllamaBenchmarkWorker(client, request.model, request.prompt, num_ctx=request.context_limit,
                              num_predict=request.output_limit, num_thread=request.cpu_threads,
                              temperature=request.temperature, keep_alive=request.keep_alive).run()
        client.generate.assert_called_once_with(request.model, request.prompt, num_ctx=4096,
                                                num_predict=32, num_thread=4, temperature=0, keep_alive=0)

    def test_allow_runs_and_passes_with_cleanup_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, resources = self.run_engine(directory)
        record = next(iter(result.records.values()))
        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertTrue(record.cleanup_verified)
        self.assertEqual(provider.execute_calls, [record.candidate_id])
        self.assertGreaterEqual(len(resources.calls), 1)

    def test_delay_waits_reassesses_fresh_and_does_not_eliminate(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, resources = self.run_engine(
                directory, decisions=(ResourceDecision.DELAY, ResourceDecision.ALLOW))
        record = next(iter(result.records.values()))
        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertGreaterEqual(len(resources.calls), 2)
        self.assertEqual(provider.execute_calls, [record.candidate_id])

    def test_static_deny_eliminates_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, _ = self.run_engine(directory, decisions=(ResourceDecision.DENY,))
        record = next(iter(result.records.values()))
        self.assertIs(record.outcome, SurvivalOutcome.ELIMINATED)
        self.assertIn("RESOURCE_DENIED", record.reason)
        self.assertEqual(provider.execute_calls, [])

    def test_candidates_execute_sequentially_in_catalog_order(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, _ = self.run_engine(directory, count=2)
        self.assertEqual(provider.execute_calls, list(result.records))
        self.assertEqual(provider.maximum_concurrent, 1)

    def test_cleanup_failure_is_hard_elimination(self):
        failed_cleanup = SurvivalExecutionResult(benchmark(), {}, False, True, True)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, _ = self.run_engine(directory, executions=(failed_cleanup,))
        record = next(iter(result.records.values()))
        self.assertIs(record.outcome, SurvivalOutcome.ELIMINATED)
        self.assertIn("CLEANUP_FAILED", record.reason)

    def test_timeout_still_requires_cleanup_and_records_timeout_kind(self):
        timed = SurvivalExecutionResult(benchmark(BenchmarkOutcome.TIMEOUT, timeout_kind="model"),
                                        {"timeout_kind": "model"}, True, True, True)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, _ = self.run_engine(directory, executions=(timed,))
        record = next(iter(result.records.values()))
        self.assertIs(record.outcome, SurvivalOutcome.ELIMINATED)
        self.assertTrue(record.cleanup_verified)
        self.assertEqual(record.metrics["timeout_kind"], "model")

    def test_thermal_abort_cleans_up_and_eliminates(self):
        thermal = SurvivalExecutionResult(benchmark(BenchmarkOutcome.THERMAL_LIMIT), {}, True, True, True)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, _ = self.run_engine(directory, executions=(thermal,))
        record = next(iter(result.records.values()))
        self.assertIn("THERMAL_ABORT", record.reason)
        self.assertTrue(record.cleanup_verified)

    def test_default_identity_failure_is_hard_elimination(self):
        isolated = SurvivalExecutionResult(benchmark(), {}, True, True, False)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, _ = self.run_engine(directory, executions=(isolated,))
        self.assertIn("POLICY_VIOLATION", next(iter(result.records.values())).reason)

    def test_persistence_round_trip_and_policy_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, _, policy = self.create(directory, allow=True)
            self.assertEqual(store.load(campaign.tournament_id), campaign)
            self.assertTrue(store.load_policy(campaign.tournament_id).allow_model_download)
            self.assertEqual(survival_campaign_from_dict(survival_campaign_to_dict(campaign)), campaign)

    def test_verified_provisioning_evidence_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            candidate = manifest.candidates[0]
            record = replace(campaign.records[candidate.candidate_id],
                             provisioning_state=ProvisioningState.VERIFIED,
                             provisioning=evidence(candidate))
            saved = store.save(replace(campaign, records={candidate.candidate_id: record},
                                       updated_at=NOW + timedelta(seconds=1)))
            altered = replace(record, provisioning=evidence(candidate, digest="sha256:rewritten"))
            with self.assertRaisesRegex(RuntimeError, "immutable"):
                store.save(replace(saved, records={candidate.candidate_id: altered},
                                   updated_at=NOW + timedelta(seconds=2)))

    def test_completed_candidate_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, policy = self.create(directory)
            candidate = manifest.candidates[0]
            passed = replace(campaign.records[candidate.candidate_id],
                             provisioning_state=ProvisioningState.VERIFIED,
                             provisioning=evidence(candidate), outcome=SurvivalOutcome.PASSED,
                             cleanup_verified=True, resource_recovery_verified=True,
                             default_server_identity_preserved=True)
            store.save(replace(campaign, records={candidate.candidate_id: passed},
                               updated_at=NOW + timedelta(seconds=1)))
            provider = FakeProvider(manifest, FakeResourceManager(), policy)
            clock = Clock()
            result = SurvivalTournamentEngine(campaign.tournament_id, store, provider,
                                              now=clock.now, sleep=clock.sleep).run()
            self.assertIs(result.state, SurvivalCampaignState.COMPLETED)
            self.assertEqual(provider.execute_calls, [])

    def test_report_preserves_missing_metrics_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            payload = survival_report_payload(campaign, manifest)
        self.assertIsNone(payload["candidates"][0]["cold_load_latency_seconds"])
        self.assertEqual(payload["summary"]["PENDING"], 1)

    def test_status_detects_stale_supervisor_and_recovers_running_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            candidate = manifest.candidates[0]
            running = replace(campaign.records[candidate.candidate_id], outcome=SurvivalOutcome.RUNNING,
                              attempt_count=1)
            store.save(replace(campaign, state=SurvivalCampaignState.RUNNING,
                               records={candidate.candidate_id: running}, updated_at=NOW + timedelta(seconds=1)))
            manager = TournamentSupervisorManager(store)
            manager.process_matches = lambda pid, started: False
            recovered, alive = manager.status(campaign.tournament_id)
        self.assertFalse(alive)
        self.assertIs(recovered.state, SurvivalCampaignState.INTERRUPTED)
        self.assertIs(recovered.records[candidate.candidate_id].outcome, SurvivalOutcome.INTERRUPTED)

    def test_one_live_tournament_blocks_second_start(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, _, _ = self.create(directory)
            store.save(replace(campaign, state=SurvivalCampaignState.RUNNING,
                               updated_at=NOW + timedelta(seconds=1), supervisor_pid=99,
                               supervisor_started_epoch=1.0))
            manager = TournamentSupervisorManager(store)
            manager.process_matches = lambda pid, started: True
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                manager.start(detach=True)

    def test_watch_snapshot_is_read_only_and_telemetry_failure_is_graceful(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, _, _ = self.create(directory)
            before = store.execution_path(campaign.tournament_id).read_bytes()
            resource_manager = Mock()
            resource_manager.current_budget.side_effect = RuntimeError("telemetry unavailable")
            watcher = TournamentWatcher(Path(directory), resource_manager=resource_manager)
            snapshot = watcher.snapshot()
            rendered = TournamentWatchRenderer().render(snapshot)
            after = store.execution_path(campaign.tournament_id).read_bytes()
        self.assertEqual(before, after)
        self.assertIn("Telemetry unavailable", rendered)
        self.assertIn("ROUND 1 RESULTS", rendered)

    def test_watch_ctrl_c_exits_viewer_without_cancellation(self):
        fake = Mock()
        fake.snapshot.side_effect = KeyboardInterrupt
        output = io.StringIO()
        self.assertEqual(watch(Path("unused"), watcher=fake, output=output), 0)

    def test_read_only_cli_commands_are_inert(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, _, _ = self.create(directory)
            with patch.object(OllamaClient, "pull", side_effect=AssertionError("pull")) as pull, \
                 patch.object(OllamaClient, "generate", side_effect=AssertionError("generate")) as generate, \
                 patch.object(OwnedOllamaServer, "start", side_effect=AssertionError("start")) as start, \
                 patch.object(OwnedOllamaServer, "shutdown", side_effect=AssertionError("shutdown")) as shutdown:
                for command in (("list",), ("status", campaign.tournament_id),
                                ("report", campaign.tournament_id), ("manifest", campaign.tournament_id)):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(tournament_main(("--root", directory, *command)), 0)
                pull.assert_not_called()
                generate.assert_not_called()
                start.assert_not_called()
                shutdown.assert_not_called()

    def test_source_contains_no_process_name_killing(self):
        source = (Path(__file__).parents[1] / "src" / "ttea" / "tournament" / "supervisor.py").read_text(encoding="utf-8")
        self.assertNotIn("taskkill", source.lower())
        self.assertNotIn("killall", source.lower())


    def test_selected_gpu_index_reaches_execution_request(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, policy = self.create(directory)
            resources = FakeResourceManager(selected_gpu_index=7)
            provider = FakeProvider(manifest, resources, policy)
            result = SurvivalTournamentEngine(
                campaign.tournament_id,
                store,
                provider,
                now=Clock().now,
                sleep=lambda _: None,
            ).run()
            self.assertIs(result.state, SurvivalCampaignState.COMPLETED)
            self.assertEqual(len(provider.execute_requests), 1)
            self.assertEqual(provider.execute_requests[0].gpu_index, 7)

    def test_survival_store_blocks_terminal_reactivation_and_generic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, _, _ = self.create(directory)

            failed = store.save(
                replace(
                    campaign,
                    state=SurvivalCampaignState.FAILED,
                    updated_at=NOW + timedelta(seconds=1),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "terminal|illegal"):
                store.save(
                    replace(
                        failed,
                        state=SurvivalCampaignState.RUNNING,
                        updated_at=NOW + timedelta(seconds=2),
                    )
                )

        with tempfile.TemporaryDirectory() as directory:
            store, campaign, _, _ = self.create(directory)
            interrupted = store.save(
                replace(
                    campaign,
                    state=SurvivalCampaignState.INTERRUPTED,
                    updated_at=NOW + timedelta(seconds=1),
                )
            )
            resumed = replace(
                interrupted,
                state=SurvivalCampaignState.CREATED,
                updated_at=NOW + timedelta(seconds=2),
            )
            with self.assertRaisesRegex(RuntimeError, "illegal"):
                store.save(resumed)

            explicit = store.save(resumed, allow_resume=True)
            self.assertIs(explicit.state, SurvivalCampaignState.CREATED)

    def test_real_provider_survival_request_does_not_use_tier_vram_as_hard_requirement(self):
        manifest = build_survival_manifest()
        candidate = manifest.candidates[-1]

        provider = object.__new__(OllamaTournamentProvider)
        provider.manifest = manifest
        provider._allowed = {
            item.candidate_id: item.variant.model for item in manifest.candidates
        }

        request = provider.resource_request(
            candidate,
            manifest.rounds[0],
            manifest.tasks[0],
        )

        self.assertEqual(request.estimated_gpu_vram_bytes, 256 * 1024 * 1024)
        self.assertEqual(
            request.estimated_ram_bytes,
            512 * 1024 * 1024,
        )
        self.assertLess(
            request.estimated_gpu_vram_bytes,
            candidate.estimated_vram_bytes,
        )


    def test_transient_owned_runtime_startup_retries_without_elimination(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, policy = self.create(directory)
            resources = FakeResourceManager()
            provider = FakeProvider(manifest, resources, policy)
            original_provision = provider.provision
            calls = {"count": 0}

            def flaky_provision(candidate, prior=None, acquisition_preflight=None):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise ProvisioningRuntimeDelayed(
                        "owned Ollama server startup failed: "
                        "owned Ollama server startup timed out"
                    )
                return original_provision(
                    candidate,
                    prior,
                    acquisition_preflight,
                )

            provider.provision = flaky_provision
            clock = Clock()

            result = SurvivalTournamentEngine(
                campaign.tournament_id,
                store,
                provider,
                now=clock.now,
                sleep=clock.sleep,
            ).run()

            record = next(iter(result.records.values()))
            self.assertIs(result.state, SurvivalCampaignState.COMPLETED)
            self.assertIs(record.outcome, SurvivalOutcome.PASSED)
            self.assertEqual(calls["count"], 2)
            self.assertNotIn("RUNTIME_FAILED", record.reason or "")


    def test_qwen35_catalog_quantization_and_explicit_q4_tag(self):
        manifest = build_survival_manifest()
        candidates = {
            candidate.candidate_id: candidate
            for candidate in manifest.candidates
        }

        qwen08 = candidates["qwen35-08b"]
        self.assertEqual(qwen08.variant.model, "qwen3.5:0.8b")
        self.assertEqual(qwen08.variant.quantization, "Q8_0")
        self.assertEqual(qwen08.estimated_disk_bytes, 1_000_000_000)

        qwen2 = candidates["qwen35-2b"]
        self.assertEqual(
            qwen2.variant.model,
            "qwen3.5:2b-q4_K_M",
        )
        self.assertEqual(qwen2.variant.quantization, "Q4_K_M")
        self.assertEqual(qwen2.estimated_disk_bytes, 1_900_000_000)

    def test_metadata_mismatch_blocks_without_model_elimination(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, policy = self.create(directory)
            resources = FakeResourceManager()
            provider = FakeProvider(manifest, resources, policy)

            def reject_metadata(
                candidate,
                prior=None,
                acquisition_preflight=None,
            ):
                raise ProvisioningRejected(
                    "installed model quantization does not match manifest"
                )

            provider.provision = reject_metadata
            clock = Clock()

            result = SurvivalTournamentEngine(
                campaign.tournament_id,
                store,
                provider,
                now=clock.now,
                sleep=clock.sleep,
            ).run()

            record = next(iter(result.records.values()))

            self.assertIs(record.outcome, SurvivalOutcome.BLOCKED)
            self.assertIs(
                record.provisioning_state,
                ProvisioningState.BLOCKED,
            )
            self.assertIn("CONFIGURATION_MISMATCH", record.reason)
            self.assertEqual(provider.execute_calls, [])

    def test_resource_retry_does_not_reprovision_verified_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, resources = self.run_engine(
                directory,
                decisions=(
                    ResourceDecision.DELAY,
                    ResourceDecision.ALLOW,
                ),
            )

        record = next(iter(result.records.values()))

        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertEqual(len(provider.provision_calls), 1)
        self.assertGreaterEqual(len(resources.calls), 2)

    def test_real_survival_request_uses_bounded_startup_headroom(self):
        manifest = self.manifest()
        candidate = manifest.candidates[0]
        provider = self.provider(manifest)

        request = provider.resource_request(
            candidate,
            manifest.rounds[0],
            manifest.tasks[0],
        )

        self.assertEqual(
            request.estimated_ram_bytes,
            512 * 1024 * 1024,
        )
        self.assertEqual(request.estimated_disk_bytes, 0)
        self.assertEqual(
            request.estimated_gpu_vram_bytes,
            256 * 1024 * 1024,
        )
        self.assertLess(
            request.estimated_ram_bytes,
            candidate.estimated_disk_bytes,
        )


if __name__ == "__main__":
    unittest.main()
