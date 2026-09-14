from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from ttea.benchmarking import BenchmarkOutcome, BenchmarkResult, OllamaBenchmarkWorker, OllamaClient, OwnedOllamaServer
from ttea.benchmarking.ollama import OllamaRuntimeError
from ttea.resources import ResourceDecision, ResourceRequest
from ttea.tournament import (
    ActiveSurvivalAttempt, AttemptEvidenceState, CandidateSurvivalRecord,
    CandidateVariant, OllamaTournamentProvider,
    ProvisioningDeferred, ProvisioningEvidence, ProvisioningRejected,
    ProvisioningRuntimeDelayed, ProvisioningState, ResourceAdmissionEvidence,
    SurvivalAttemptEvidence,
    SurvivalCampaign, SurvivalCampaignState,
    SurvivalExecutionPolicy, SurvivalExecutionResult, SurvivalOutcome,
    SurvivalStore, SurvivalTournamentEngine, TournamentCandidate,
    TournamentManifest, TournamentStoreError, TournamentSupervisorManager, build_survival_manifest,
    survival_campaign_from_dict, survival_campaign_to_dict,
    survival_attempt_id, survival_report_payload, write_survival_report,
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
                execution = self.executions.pop(0)
                if isinstance(execution, BaseException):
                    raise execution
                return execution
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

    def create(self, directory, *, count=1, allow=False, policy=None,
               tournament_id="survival-test"):
        manifest = self.manifest(count)
        policy = policy or SurvivalExecutionPolicy(allow_model_download=allow, resource_retry_seconds=0.5)
        campaign = SurvivalCampaign(
            tournament_id, manifest.digest, SurvivalCampaignState.CREATED, NOW, NOW,
            NOW + timedelta(hours=6),
            {item.candidate_id: CandidateSurvivalRecord(item.candidate_id, item.variant.model)
             for item in manifest.candidates},
        )
        store = SurvivalStore(directory)
        store.create(campaign, manifest, policy)
        return store, campaign, manifest, policy

    def admitted_attempt(self, campaign, candidate_id, *, attempt_number=1):
        request = ResourceRequest(
            workload_type=f"tournament-survival:{candidate_id}",
            estimated_ram_bytes=1,
            estimated_disk_bytes=0,
            estimated_gpu_vram_bytes=1,
            id=f"request:{candidate_id}:{attempt_number}",
        )
        return ActiveSurvivalAttempt(
            survival_attempt_id(campaign.tournament_id, candidate_id, attempt_number),
            campaign.tournament_id,
            candidate_id,
            attempt_number,
            "survival_4k",
            "survival_exact_reply",
            NOW,
            request,
            ResourceAdmissionEvidence(
                request.id,
                ResourceDecision.ALLOW,
                ("resource allow",),
            ),
        )

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

    def test_owned_server_identity_is_observed_when_runtime_ownership_is_recorded(self):
        provider = self.provider(self.manifest())
        server = Mock()
        provider.server_factory = Mock(return_value=server)
        state_store = Mock()
        observed = []
        self.assertIs(
            provider._server(provider.benchmark_policy(), state_store, observed.append),
            server,
        )
        identity = SimpleNamespace(pid=4321, started_epoch=87.5)
        callback = provider.server_factory.call_args.kwargs["identity_callback"]
        callback(identity)
        state_store.record_server.assert_called_once_with(4321, 87.5)
        self.assertEqual(observed, [identity])

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
            result, store, provider, resources = self.run_engine(directory)
            record = next(iter(result.records.values()))
            attempts = store.attempts_for(result.tournament_id, record.candidate_id)
        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertTrue(record.cleanup_verified)
        self.assertEqual(provider.execute_calls, [record.candidate_id])
        self.assertGreaterEqual(len(resources.calls), 1)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].attempt.attempt_number, 1)
        self.assertIs(attempts[0].benchmark_outcome, BenchmarkOutcome.COMPLETED)

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
            result, store, provider, _ = self.run_engine(directory, count=2)
            attempts = store.attempts_by_candidate(result.tournament_id)
        self.assertEqual(provider.execute_calls, list(result.records))
        self.assertEqual(provider.maximum_concurrent, 1)
        self.assertEqual(set(attempts), set(result.records))
        for candidate_id, candidate_attempts in attempts.items():
            self.assertEqual(len(candidate_attempts), 1)
            self.assertEqual(candidate_attempts[0].attempt.candidate_id, candidate_id)

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

    def test_runtime_resource_pressure_then_success_is_bounded_and_reuses_provisioning(self):
        pressure = SurvivalExecutionResult(
            benchmark(BenchmarkOutcome.RESOURCE_PRESSURE),
            {"ram_minimum_available_bytes": 100}, True, True, True,
        )
        success = SurvivalExecutionResult(
            benchmark(), {"ram_minimum_available_bytes": 200}, True, True, True,
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, _ = self.run_engine(
                directory, executions=(pressure, success),
            )
        record = next(iter(result.records.values()))
        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(len(provider.execute_calls), 2)
        self.assertEqual(len(provider.provision_calls), 1)

    def test_repeated_runtime_resource_pressure_eliminates_at_default_bound(self):
        first = SurvivalExecutionResult(
            BenchmarkResult(
                BenchmarkOutcome.RESOURCE_PRESSURE,
                ("available RAM remained below the reserved RAM threshold",),
                NOW, NOW + timedelta(seconds=1), 321, None, True,
            ),
            {"sample": 1, "ram_minimum_available_bytes": 111}, True, True, True,
            9001, 10.5,
        )
        second = SurvivalExecutionResult(
            BenchmarkResult(
                BenchmarkOutcome.RESOURCE_PRESSURE,
                ("runtime RAM reserve crossed again",),
                NOW, NOW + timedelta(seconds=1), 654, None, True,
            ),
            {"sample": 2, "ram_minimum_available_bytes": 222}, True, True, True,
            9002, 20.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            result, store, provider, _ = self.run_engine(
                directory, executions=(first, second),
            )
            record = next(iter(result.records.values()))
            attempts = store.attempts_for(result.tournament_id, record.candidate_id)
            report = json.loads(
                store.tournaments.report_json_path(result.tournament_id).read_text(encoding="utf-8")
            )
        self.assertIs(record.outcome, SurvivalOutcome.ELIMINATED)
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(len(provider.execute_calls), 2)
        self.assertEqual(len(provider.provision_calls), 1)
        self.assertIn("RESOURCE_DENIED", record.reason)
        self.assertIn("candidate/runtime configuration", record.reason)
        self.assertIn("runtime RAM reserve crossed again", record.reason)
        self.assertEqual(record.metrics["sample"], 2)
        self.assertEqual(record.worker_pid, 654)
        self.assertTrue(record.cleanup_verified)
        self.assertTrue(record.resource_recovery_verified)
        self.assertTrue(record.default_server_identity_preserved)
        self.assertEqual([item.attempt.attempt_number for item in attempts], [1, 2])
        self.assertEqual([item.metrics["ram_minimum_available_bytes"] for item in attempts], [111, 222])
        self.assertEqual([item.worker_pid for item in attempts], [321, 654])
        self.assertEqual([item.owned_ollama_pid for item in attempts], [9001, 9002])
        self.assertEqual([item.owned_ollama_started_epoch for item in attempts], [10.5, 20.5])
        self.assertTrue(all(item.cleanup_verified for item in attempts))
        self.assertTrue(all(item.resource_recovery_verified for item in attempts))
        self.assertTrue(all(item.default_server_identity_preserved for item in attempts))
        self.assertEqual(report["candidates"][0]["survival_outcome"], record.outcome.value)
        self.assertEqual(
            len(report["attempt_evidence"]["by_candidate"][record.candidate_id]),
            2,
        )

    def test_repeated_pressure_eliminates_first_then_scheduler_runs_second(self):
        pressure = SurvivalExecutionResult(
            benchmark(BenchmarkOutcome.RESOURCE_PRESSURE), {}, True, True, True,
        )
        success = SurvivalExecutionResult(benchmark(), {}, True, True, True)
        with tempfile.TemporaryDirectory() as directory:
            result, _, provider, _ = self.run_engine(
                directory, count=2, executions=(pressure, pressure, success),
            )
        records = tuple(result.records.values())
        self.assertIs(records[0].outcome, SurvivalOutcome.ELIMINATED)
        self.assertIn("RESOURCE_DENIED", records[0].reason)
        self.assertIs(records[1].outcome, SurvivalOutcome.PASSED)
        self.assertEqual(provider.execute_calls,
                         [records[0].candidate_id, records[0].candidate_id, records[1].candidate_id])

    def test_pre_execution_delay_does_not_consume_runtime_pressure_attempt(self):
        pressure = SurvivalExecutionResult(
            benchmark(BenchmarkOutcome.RESOURCE_PRESSURE), {}, True, True, True,
        )
        success = SurvivalExecutionResult(benchmark(), {}, True, True, True)
        with tempfile.TemporaryDirectory() as directory:
            result, store, provider, resources = self.run_engine(
                directory,
                decisions=(ResourceDecision.DELAY, ResourceDecision.ALLOW, ResourceDecision.ALLOW),
                executions=(pressure, success),
            )
            record = next(iter(result.records.values()))
            attempts = store.attempts_for(result.tournament_id, record.candidate_id)
        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(len(provider.execute_calls), 2)
        self.assertEqual(len(provider.provision_calls), 1)
        self.assertEqual(len(resources.calls), 3)
        self.assertEqual([item.attempt.attempt_number for item in attempts], [1, 2])

    def test_worker_failure_retry_semantics_remain_two_attempts(self):
        failure = SurvivalExecutionResult(
            BenchmarkResult(BenchmarkOutcome.WORKER_FAILURE, ("worker failed",), NOW,
                            NOW + timedelta(seconds=1), 401, None, True),
            {"sample": "failure"}, True, True, True,
        )
        success = SurvivalExecutionResult(
            BenchmarkResult(BenchmarkOutcome.COMPLETED, ("completed",), NOW,
                            NOW + timedelta(seconds=1), 402, None, True),
            {"sample": "success"}, True, True, True,
        )
        with tempfile.TemporaryDirectory() as directory:
            result, store, provider, _ = self.run_engine(
                directory, executions=(failure, success),
            )
            record = next(iter(result.records.values()))
            attempts = store.attempts_for(result.tournament_id, record.candidate_id)
        self.assertIs(record.outcome, SurvivalOutcome.PASSED)
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(len(provider.execute_calls), 2)
        self.assertEqual(len(provider.provision_calls), 1)
        self.assertEqual([item.worker_pid for item in attempts], [401, 402])
        self.assertEqual([item.metrics["sample"] for item in attempts], ["failure", "success"])
        self.assertEqual(
            [item.benchmark_outcome for item in attempts],
            [BenchmarkOutcome.WORKER_FAILURE, BenchmarkOutcome.COMPLETED],
        )

    def test_delay_cycles_create_no_evidence_until_first_admitted_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, policy = self.create(directory)
            observed_counts = []

            class ObservingResources(FakeResourceManager):
                def assess(inner_self, request):
                    assessment = super().assess(request)
                    if assessment.decision is ResourceDecision.DELAY:
                        observed_counts.append(len(store.attempts_for(
                            campaign.tournament_id,
                            manifest.candidates[0].candidate_id,
                        )))
                    return assessment

            resources = ObservingResources((
                ResourceDecision.DELAY,
                ResourceDecision.DELAY,
                ResourceDecision.ALLOW,
            ))
            provider = FakeProvider(manifest, resources, policy)
            clock = Clock()
            result = SurvivalTournamentEngine(
                campaign.tournament_id,
                store,
                provider,
                now=clock.now,
                sleep=clock.sleep,
            ).run()
            candidate_id = manifest.candidates[0].candidate_id
            attempts = store.attempts_for(result.tournament_id, candidate_id)

        self.assertEqual(observed_counts, [0, 0])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].attempt.attempt_number, 1)

    def test_attempt_preserves_admission_identity_metrics_and_cleanup_details(self):
        metric_bundle = {
            "ram_available_before_bytes": 900,
            "ram_minimum_available_bytes": 700,
            "ram_transient_delta_bytes": 200,
            "gpu_free_vram_before_bytes": 600,
            "gpu_minimum_free_vram_bytes": 500,
            "gpu_transient_vram_delta_bytes": 100,
            "max_gpu_temperature_c": 61.5,
            "wall_clock_duration_seconds": 1.0,
            "decode_tokens_per_second": 12.5,
            "response_is_string": True,
            "model_unload_confirmed": True,
            "server_shutdown_confirmed": True,
            "dedicated_port_closed": True,
            "runtime_state_absent": True,
        }

        class RichResources(FakeResourceManager):
            def assess(inner_self, request):
                inner_self.calls.append(request)
                return SimpleNamespace(
                    decision=ResourceDecision.ALLOW,
                    reasons=("resource allow",),
                    snapshot_timestamp=NOW,
                    ram_required_bytes=1,
                    ram_available_bytes=900,
                    ram_reserved_bytes=100,
                    ram_headroom_bytes=799,
                    disk_required_bytes=0,
                    disk_free_bytes=800,
                    disk_reserved_bytes=100,
                    disk_headroom_bytes=700,
                    selected_gpu_index=0,
                    gpu_total_vram_bytes=1_000,
                    gpu_free_vram_bytes=600,
                    gpu_reserved_vram_bytes=100,
                    gpu_headroom_bytes=499,
                    gpu_temperature_c=55.5,
                )

        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, policy = self.create(directory)
            result_value = SurvivalExecutionResult(
                BenchmarkResult(
                    BenchmarkOutcome.COMPLETED,
                    ("completed",),
                    NOW,
                    NOW + timedelta(seconds=1),
                    7303,
                    None,
                    True,
                    None,
                    13.5,
                ),
                metric_bundle,
                True,
                True,
                True,
                7404,
                14.5,
            )
            provider = FakeProvider(manifest, RichResources(), policy, (result_value,))
            result = SurvivalTournamentEngine(
                campaign.tournament_id,
                store,
                provider,
                now=Clock().now,
                sleep=lambda _seconds: None,
            ).run()
            candidate_id = manifest.candidates[0].candidate_id
            attempt = store.attempts_for(result.tournament_id, candidate_id)[0]

        self.assertEqual(dict(attempt.metrics), metric_bundle)
        self.assertEqual(attempt.attempt.resource_request, provider.execute_requests[0])
        self.assertIs(attempt.attempt.admission.decision, ResourceDecision.ALLOW)
        self.assertEqual(attempt.attempt.admission.snapshot_timestamp, NOW)
        self.assertEqual(attempt.attempt.admission.ram_available_bytes, 900)
        self.assertEqual(attempt.attempt.admission.gpu_temperature_c, 55.5)
        self.assertEqual((attempt.worker_pid, attempt.worker_started_epoch), (7303, 13.5))
        self.assertEqual((attempt.owned_ollama_pid, attempt.owned_ollama_started_epoch), (7404, 14.5))
        self.assertTrue(attempt.response_valid)
        self.assertTrue(attempt.cleanup_verified)
        self.assertTrue(attempt.resource_recovery_verified)
        self.assertTrue(attempt.default_server_identity_preserved)

    def test_interrupted_execution_finalizes_the_admitted_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            result, store, _, _ = self.run_engine(
                directory,
                executions=(KeyboardInterrupt(),),
            )
            candidate_id = next(iter(result.records))
            attempts = store.attempts_for(result.tournament_id, candidate_id)

        self.assertIs(result.state, SurvivalCampaignState.INTERRUPTED)
        self.assertIsNone(result.active_attempt)
        self.assertEqual(len(attempts), 1)
        self.assertIs(attempts[0].evidence_state, AttemptEvidenceState.INTERRUPTED)
        self.assertEqual(attempts[0].attempt.attempt_number, 1)
        self.assertIsNone(attempts[0].benchmark_outcome)

    def test_recovery_captures_owned_identities_and_does_not_duplicate_finalized_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            candidate_id = manifest.candidates[0].candidate_id
            active = self.admitted_attempt(campaign, candidate_id)
            running_record = replace(
                campaign.records[candidate_id],
                outcome=SurvivalOutcome.RUNNING,
                attempt_count=1,
            )
            running = store.save(replace(
                campaign,
                state=SurvivalCampaignState.RUNNING,
                records={candidate_id: running_record},
                current_candidate_id=candidate_id,
                current_round_id=active.round_id,
                updated_at=NOW + timedelta(seconds=1),
                active_attempt=active,
            ))
            recovered = store.recover_active_attempt(
                running,
                finished_at=NOW + timedelta(seconds=2),
                ownership_state={
                    "worker_pid": 7101,
                    "worker_started_epoch": 11.5,
                    "server_pid": 7202,
                    "server_started_epoch": 12.5,
                },
            )
            store.save(replace(recovered, updated_at=NOW + timedelta(seconds=2)))
            first = store.attempts_for(campaign.tournament_id, candidate_id)
            recovered_again = store.recover_active_attempt(
                running,
                finished_at=NOW + timedelta(seconds=3),
            )
            second = store.attempts_for(campaign.tournament_id, candidate_id)

        self.assertIsNone(recovered.active_attempt)
        self.assertIsNone(recovered_again.active_attempt)
        self.assertEqual(first, second)
        self.assertEqual(len(second), 1)
        self.assertEqual((second[0].worker_pid, second[0].worker_started_epoch), (7101, 11.5))
        self.assertEqual(
            (second[0].owned_ollama_pid, second[0].owned_ollama_started_epoch),
            (7202, 12.5),
        )

    def test_status_and_resume_do_not_duplicate_an_already_finalized_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            candidate_id = manifest.candidates[0].candidate_id
            active = self.admitted_attempt(campaign, candidate_id)
            running_record = replace(
                campaign.records[candidate_id],
                outcome=SurvivalOutcome.RUNNING,
                attempt_count=1,
            )
            store.save(replace(
                campaign,
                state=SurvivalCampaignState.RUNNING,
                records={candidate_id: running_record},
                current_candidate_id=candidate_id,
                current_round_id=active.round_id,
                updated_at=NOW + timedelta(seconds=1),
                active_attempt=active,
            ))
            terminal = SurvivalAttemptEvidence(
                active,
                AttemptEvidenceState.FINALIZED,
                NOW + timedelta(seconds=2),
                BenchmarkOutcome.RESOURCE_PRESSURE,
                ("runtime pressure",),
                metrics={"ram_minimum_available_bytes": 123},
                cleanup_verified=True,
                resource_recovery_verified=True,
                default_server_identity_preserved=True,
            )
            store.finalize_attempt(terminal)
            manager = TournamentSupervisorManager(
                store,
                benchmark_state_path=Path(directory) / "benchmark.json",
                process_api=lambda _pid: SimpleNamespace(create_time=lambda: 1.0),
            )
            manager.process_matches = lambda _pid, _started: False
            manager._spawn = lambda value: value
            manager.status(campaign.tournament_id)
            manager.status(campaign.tournament_id)
            resumed = manager.resume(campaign.tournament_id)
            attempts = store.attempts_for(campaign.tournament_id, candidate_id)

        self.assertIs(resumed.state, SurvivalCampaignState.CREATED)
        self.assertEqual(attempts, (terminal,))

    def test_campaign_attempt_evidence_isolated_for_same_candidate_id(self):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for tournament_id, minimum in (("campaign-a", 101), ("campaign-b", 202)):
                store, campaign, manifest, policy = self.create(
                    directory,
                    tournament_id=tournament_id,
                )
                provider = FakeProvider(
                    manifest,
                    FakeResourceManager(),
                    policy,
                    executions=(SurvivalExecutionResult(
                        benchmark(),
                        {"ram_minimum_available_bytes": minimum},
                        True,
                        True,
                        True,
                    ),),
                )
                result = SurvivalTournamentEngine(
                    campaign.tournament_id,
                    store,
                    provider,
                    now=Clock().now,
                    sleep=lambda _seconds: None,
                ).run()
                candidate_id = manifest.candidates[0].candidate_id
                results.append(store.attempts_for(result.tournament_id, candidate_id))

        self.assertEqual(results[0][0].metrics["ram_minimum_available_bytes"], 101)
        self.assertEqual(results[1][0].metrics["ram_minimum_available_bytes"], 202)
        self.assertNotEqual(results[0][0].attempt.tournament_id, results[1][0].attempt.tournament_id)

    def test_historical_campaign_without_attempt_schema_loads_and_reports_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            store.attempts_path(campaign.tournament_id).rmdir()
            execution_path = store.execution_path(campaign.tournament_id)
            raw = json.loads(execution_path.read_text(encoding="utf-8"))
            raw.pop("active_attempt")
            execution_path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = store.load(campaign.tournament_id)
            self.assertFalse(store.attempt_evidence_available(campaign.tournament_id))
            self.assertEqual(
                store.attempts_for(campaign.tournament_id, manifest.candidates[0].candidate_id),
                (),
            )
            write_survival_report(store, loaded, manifest)
            report = json.loads(
                store.tournaments.report_json_path(campaign.tournament_id).read_text(encoding="utf-8")
            )
            watcher = TournamentWatcher(Path(directory), resource_manager=Mock())
            rendered_watch = TournamentWatchRenderer().render(watcher.snapshot())

        self.assertIsNone(loaded.active_attempt)
        self.assertFalse(report["attempt_evidence"]["available"])
        self.assertEqual(report["attempt_evidence"]["by_candidate"][manifest.candidates[0].candidate_id], [])
        self.assertIn("ROUND 1 RESULTS", rendered_watch)

    def test_finalized_attempt_publication_is_idempotent_but_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store, campaign, manifest, _ = self.create(directory)
            candidate_id = manifest.candidates[0].candidate_id
            active = self.admitted_attempt(campaign, candidate_id)
            running_record = replace(
                campaign.records[candidate_id],
                outcome=SurvivalOutcome.RUNNING,
                attempt_count=1,
            )
            store.save(replace(
                campaign,
                state=SurvivalCampaignState.RUNNING,
                records={candidate_id: running_record},
                current_candidate_id=candidate_id,
                current_round_id=active.round_id,
                updated_at=NOW + timedelta(seconds=1),
                active_attempt=active,
            ))
            terminal = SurvivalAttemptEvidence(
                active,
                AttemptEvidenceState.FINALIZED,
                NOW + timedelta(seconds=2),
                BenchmarkOutcome.COMPLETED,
                ("completed",),
                metrics={"ram_minimum_available_bytes": 111},
                cleanup_verified=True,
                resource_recovery_verified=True,
                default_server_identity_preserved=True,
            )
            store.finalize_attempt(terminal)
            self.assertEqual(store.finalize_attempt(terminal), terminal)
            with self.assertRaisesRegex(TournamentStoreError, "immutable attempt evidence conflict"):
                store.finalize_attempt(replace(
                    terminal,
                    metrics={"ram_minimum_available_bytes": 999},
                ))

    def test_attempt_models_reject_malformed_identity_state_timestamps_and_process_pairs(self):
        manifest = self.manifest()
        candidate_id = manifest.candidates[0].candidate_id
        campaign = SurvivalCampaign(
            "validation-test", manifest.digest, SurvivalCampaignState.CREATED, NOW, NOW,
            NOW + timedelta(hours=1),
            {candidate_id: CandidateSurvivalRecord(candidate_id, manifest.candidates[0].variant.model)},
        )
        active = self.admitted_attempt(campaign, candidate_id)
        terminal = SurvivalAttemptEvidence(
            active,
            AttemptEvidenceState.FINALIZED,
            NOW + timedelta(seconds=1),
            BenchmarkOutcome.COMPLETED,
            ("completed",),
        )

        with self.assertRaises(ValueError):
            replace(active, attempt_number=0)
        with self.assertRaises(ValueError):
            replace(active, attempt_id="wrong-attempt-identity")
        with self.assertRaises(ValueError):
            replace(
                campaign,
                state=SurvivalCampaignState.RUNNING,
                records={candidate_id: replace(
                    campaign.records[candidate_id],
                    outcome=SurvivalOutcome.RUNNING,
                    attempt_count=1,
                )},
                current_candidate_id="another-candidate",
                current_round_id=active.round_id,
                active_attempt=active,
            )
        with self.assertRaises(ValueError):
            replace(terminal, finished_at=NOW - timedelta(seconds=1))
        with self.assertRaises(ValueError):
            replace(terminal, worker_started_epoch=1.0)
        with self.assertRaises(ValueError):
            replace(terminal, worker_pid=0)
        with self.assertRaises(ValueError):
            replace(terminal, worker_pid=123, worker_started_epoch=float("nan"))

    def test_nonfinite_attempt_metric_is_rejected_before_persistence(self):
        manifest = self.manifest()
        candidate_id = manifest.candidates[0].candidate_id
        campaign = SurvivalCampaign(
            "validation-test", manifest.digest, SurvivalCampaignState.CREATED, NOW, NOW,
            NOW + timedelta(hours=1),
            {candidate_id: CandidateSurvivalRecord(candidate_id, manifest.candidates[0].variant.model)},
        )
        active = self.admitted_attempt(campaign, candidate_id)
        with self.assertRaises(ValueError):
            SurvivalAttemptEvidence(
                active,
                AttemptEvidenceState.FINALIZED,
                NOW + timedelta(seconds=1),
                BenchmarkOutcome.COMPLETED,
                ("completed",),
                metrics={"max_gpu_temperature_c": float("nan")},
            )

    def test_generic_base_record_is_not_authoritative_survival_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            result, store, _, _ = self.run_engine(directory)
            base = store.tournaments.load(result.tournament_id)

        self.assertIs(result.state, SurvivalCampaignState.COMPLETED)
        self.assertEqual(base.state.value, "CREATED")
        self.assertEqual(base.progress_completed, 0)


if __name__ == "__main__":
    unittest.main()
