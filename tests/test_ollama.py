from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from ttea.benchmarking import BenchmarkOutcome, BenchmarkResult, BenchmarkSafetyPolicy, BenchmarkStateStore, ControlledOllamaRuntime, OllamaBenchmarkWorker, OllamaClient, OllamaEndpoint, OllamaRuntimeError, OwnedOllamaServer, resolve_ollama_executable
from ttea.benchmarking.ollama import OllamaCleanup, OllamaServerStartResult, OwnedProcessIdentity, passive_ollama_check
from ttea.resources import ResourceDecision, ResourceRequest


class HttpResponse:
    def __init__(self, payload): self.payload = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return self.payload


class FakeProcess:
    def __init__(self, pid, started=10.0, children=(), survives=False):
        self.pid, self._started, self._children, self.survives = pid, started, list(children), survives
        self.running, self.terminated, self.killed = True, False, False
    def create_time(self): return self._started
    def children(self, recursive=True): return self._children
    def terminate(self):
        self.terminated = True
        if not self.survives: self.running = False
    def kill(self):
        self.killed = True
        if not self.survives: self.running = False
    def is_running(self): return self.running


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state_path = Path(self.temporary.name) / "state.json"
        self.policy = BenchmarkSafetyPolicy()
        self.endpoint = OllamaEndpoint()
        self.client = OllamaClient(self.endpoint, self.policy, 1.5)

    def manager(self, decision=ResourceDecision.ALLOW):
        manager = Mock(policy=None)
        manager.assess.return_value = SimpleNamespace(decision=decision, reasons=("resource preflight",))
        return manager

    def runtime(self, manager=None, **kwargs):
        return ControlledOllamaRuntime(manager or self.manager(), self.policy, state_path=self.state_path, **kwargs)

    def test_endpoint_is_loopback_only(self):
        for value in ("http://127.0.0.1:11435", "http://localhost:11435", "http://[::1]:11435"):
            self.assertIsInstance(OllamaEndpoint(value), OllamaEndpoint)
        for value in ("http://example.com:11435", "http://192.168.1.2:11435", "https://127.0.0.1:11435", "http://127.0.0.1"):
            with self.assertRaises(ValueError): OllamaEndpoint(value)

    def test_controlled_environment_does_not_mutate_global_environment(self):
        before = dict(os.environ)
        environment = OwnedOllamaServer(self.endpoint, self.policy).environment()
        self.assertEqual(os.environ, before)
        self.assertEqual(environment["OLLAMA_HOST"], "127.0.0.1:11435")
        self.assertEqual({name: environment[name] for name in ("OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_QUEUE")}, {"OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_QUEUE": "1"})
        self.assertEqual(environment["OLLAMA_CONTEXT_LENGTH"], "4096")
        self.assertEqual(environment["OLLAMA_KEEP_ALIVE"], "0")
        self.assertEqual(environment["OLLAMA_NO_CLOUD"], "1")

    @patch("ttea.benchmarking.ollama.http_request.urlopen")
    def test_health_model_lists_and_explicit_http_timeout(self, urlopen):
        urlopen.side_effect = [HttpResponse({"version": "1.2"}), HttpResponse({"models": [{"name": "a"}]}), HttpResponse({"models": [{"model": "b"}]})]
        self.assertTrue(self.client.health())
        self.assertEqual(self.client.installed_models(), ("a",))
        self.assertEqual(self.client.running_models(), ("b",))
        self.assertTrue(all(call.kwargs["timeout"] == 1.5 for call in urlopen.call_args_list))

    @patch("ttea.benchmarking.ollama.http_request.urlopen", side_effect=TimeoutError("stuck"))
    def test_http_timeout_is_surfaced(self, urlopen):
        with self.assertRaises(OllamaRuntimeError): self.client.health()
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 1.5)

    @patch("ttea.benchmarking.ollama.http_request.urlopen")
    def test_generation_payload_bounds_and_metric_parsing(self, urlopen):
        urlopen.return_value = HttpResponse({"model": "local", "response": "ok", "total_duration": 9, "load_duration": 2, "prompt_eval_count": 3, "prompt_eval_duration": 4, "eval_count": 5, "eval_duration": 6, "done_reason": "stop"})
        result = self.client.generate("local", "prompt")
        outgoing = urlopen.call_args.args[0]
        payload = json.loads(outgoing.data)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], 0)
        self.assertEqual(payload["options"], {"num_ctx": 4096, "num_predict": 512, "num_thread": 4})
        self.assertEqual((result.model, result.eval_count, result.done_reason), ("local", 5, "stop"))
        self.assertEqual(urlopen.call_args.kwargs["timeout"], self.policy.request_timeout_seconds)
        with self.assertRaises(ValueError): self.client.generate("local", "prompt", num_ctx=4097)

    @patch("ttea.benchmarking.ollama.http_request.urlopen")
    def test_api_unload_uses_exact_model(self, urlopen):
        urlopen.return_value = HttpResponse({"done": True})
        self.client.unload("exact:tag")
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload, {"model": "exact:tag", "prompt": "", "stream": False, "keep_alive": 0})

    def test_bounded_pull_and_show_model_validation(self):
        responses = [HttpResponse({"status": "success"}), HttpResponse({"details": {"parameter_size": "135M"}})]
        with patch("ttea.benchmarking.ollama.http_request.urlopen", side_effect=responses) as urlopen:
            self.assertEqual(self.client.pull("exact", 12.0)["status"], "success")
            self.assertIn("details", self.client.show_model("exact", 3.0))
        pull_request = urlopen.call_args_list[0].args[0]
        self.assertEqual(json.loads(pull_request.data), {"model": "exact", "stream": False})
        self.assertEqual(urlopen.call_args_list[0].kwargs["timeout"], 12.0)
        self.assertEqual(urlopen.call_args_list[1].kwargs["timeout"], 3.0)
        with patch("ttea.benchmarking.ollama.http_request.urlopen", return_value=HttpResponse({"unexpected": True})):
            with self.assertRaises(OllamaRuntimeError):
                self.client.pull("exact", 1.0)
            with self.assertRaises(OllamaRuntimeError):
                self.client.show_model("exact")

    def test_worker_makes_exactly_one_generation_and_cannot_pull(self):
        client = Mock(spec=["generate"])
        client.generate.return_value = "result"
        worker = OllamaBenchmarkWorker(client, "exact", "prompt")
        self.assertEqual(worker.run(), "result")
        client.generate.assert_called_once_with("exact", "prompt")
        self.assertFalse(hasattr(worker, "pull"))

    @patch("ttea.benchmarking.ollama.resolve_ollama_executable", return_value="ollama-test")
    def test_owned_server_startup_success_and_timeout_cleanup(self, resolver):
        popen = Mock(pid=41, returncode=None)
        popen.poll.return_value = None
        process_api = lambda pid: FakeProcess(pid, 10)
        server = OwnedOllamaServer(self.endpoint, self.policy, process_factory=Mock(return_value=popen), process_api=process_api)
        with patch.object(server, "client") as client:
            client.return_value.health.return_value = True
            result = server.start()
        self.assertTrue(result.success)
        self.assertEqual(result.identity.pid, 41)

        timed = OwnedOllamaServer(self.endpoint, self.policy, startup_timeout_seconds=0.01, process_factory=Mock(return_value=popen), process_api=process_api)
        with patch.object(timed, "client") as client, patch.object(timed, "shutdown") as shutdown:
            client.return_value.health.side_effect = OllamaRuntimeError("not ready")
            result = timed.start()
        self.assertFalse(result.success)
        shutdown.assert_called_once()

    def test_model_installed_and_missing_model_preflight(self):
        with patch("ttea.benchmarking.ollama.http_request.urlopen", return_value=HttpResponse({"models": [{"name": "present"}]})):
            self.assertIn("present", self.client.installed_models())
        fake_client = Mock()
        fake_client.health.return_value = True
        fake_client.installed_models.return_value = ()
        fake_client.running_models.return_value = ()
        fake_server = Mock(policy=self.policy)
        fake_server.start.return_value = OllamaServerStartResult(True, ("healthy",), datetime.now(timezone.utc), datetime.now(timezone.utc), OwnedProcessIdentity(1, 1))
        fake_server.client.return_value = fake_client
        runtime = self.runtime()
        with patch("ttea.benchmarking.ollama.OwnedOllamaServer", return_value=fake_server), patch("ttea.benchmarking.ollama.BenchmarkSupervisor") as supervisor:
            result = runtime.run("missing", "prompt", ResourceRequest("test", 1, 1))
        self.assertEqual(result.outcome, BenchmarkOutcome.WORKER_FAILURE)
        self.assertIn("automatic pulls are forbidden", " ".join(result.reasons))
        self.assertNotIn("cleanup failed", " ".join(result.reasons))
        supervisor.assert_not_called()
        fake_client.unload.assert_not_called()
        fake_server.stop_model_cli.assert_not_called()
        fake_server.shutdown.assert_called_once_with(self.policy.cooldown_timeout_seconds)
        self.assertFalse(self.state_path.exists())

    def test_absent_unload_response_is_success_when_model_is_not_resident(self):
        client, server = Mock(), Mock(policy=self.policy)
        client.unload.side_effect = OllamaRuntimeError("HTTP 404")
        client.running_models.return_value = ()
        cleanup = OllamaCleanup(client, server, "absent").bounded()
        cleanup.run(0.2)
        server.stop_model_cli.assert_not_called()
        server.shutdown.assert_called_once()

    def test_unarmed_model_cleanup_is_server_only_until_generation_starts(self):
        client, server = Mock(), Mock(policy=self.policy)
        cleanup = OllamaCleanup(client, server, "present", model_cleanup_required=False)
        cleanup.bounded().run(0.2)
        client.unload.assert_not_called()
        server.stop_model_cli.assert_not_called()
        server.shutdown.assert_called_once()

    def test_controlled_runtime_routes_worker_through_supervisor(self):
        expected = BenchmarkResult(BenchmarkOutcome.COMPLETED, ("done",), datetime.now(timezone.utc), datetime.now(timezone.utc))
        fake_client = Mock()
        fake_client.health.return_value = True
        fake_client.installed_models.return_value = ("present",)
        fake_server = Mock(policy=self.policy)
        fake_server.start.return_value = OllamaServerStartResult(True, ("healthy",), datetime.now(timezone.utc), datetime.now(timezone.utc), OwnedProcessIdentity(1, 1))
        fake_server.client.return_value = fake_client
        runtime = self.runtime()
        with patch("ttea.benchmarking.ollama.OwnedOllamaServer", return_value=fake_server), patch("ttea.benchmarking.ollama.BenchmarkSupervisor") as supervisor_type:
            supervisor_type.return_value.run.return_value = expected
            result = runtime.run("present", "prompt", ResourceRequest("test", 1, 1))
        self.assertIs(result, expected)
        worker = supervisor_type.return_value.run.call_args.args[0]
        self.assertIsInstance(worker, OllamaBenchmarkWorker)

    def test_supervisor_setup_failure_still_cleans_owned_server(self):
        fake_client = Mock()
        fake_client.health.return_value = True
        fake_client.installed_models.return_value = ("present",)
        fake_client.running_models.return_value = ()
        fake_server = Mock(policy=self.policy)
        fake_server.start.return_value = OllamaServerStartResult(True, ("healthy",), datetime.now(timezone.utc), datetime.now(timezone.utc), OwnedProcessIdentity(1, 1))
        fake_server.client.return_value = fake_client
        runtime = self.runtime()
        with patch("ttea.benchmarking.ollama.OwnedOllamaServer", return_value=fake_server), patch("ttea.benchmarking.ollama.BenchmarkSupervisor", side_effect=ValueError("policy mismatch")):
            result = runtime.run("present", "prompt", ResourceRequest("test", 1, 1))
        self.assertEqual(result.outcome, BenchmarkOutcome.WORKER_FAILURE)
        fake_client.unload.assert_not_called()
        fake_server.stop_model_cli.assert_not_called()
        fake_server.shutdown.assert_called_once()

    def test_resource_delay_or_deny_never_constructs_owned_server(self):
        for decision in (ResourceDecision.DELAY, ResourceDecision.DENY):
            manager = self.manager(decision)
            with patch("ttea.benchmarking.ollama.OwnedOllamaServer") as server_type:
                result = self.runtime(manager).run("model", "prompt", ResourceRequest("test", 1, 1))
            self.assertEqual(result.outcome, BenchmarkOutcome.RESOURCE_PRESSURE)
            server_type.assert_not_called()

    def test_server_identity_is_persisted_before_worker_exists(self):
        store = BenchmarkStateStore(self.state_path)
        store.acquire()
        current_pid = os.getpid()
        current_started = psutil.Process(current_pid).create_time()
        popen = Mock(pid=current_pid, returncode=None)
        popen.poll.return_value = None
        server = OwnedOllamaServer(self.endpoint, self.policy, process_factory=Mock(return_value=popen),
                                   process_api=lambda pid: FakeProcess(pid, current_started),
                                   identity_callback=lambda identity: store.record_server(identity.pid, identity.started_epoch))
        with patch.object(server, "client") as client:
            client.return_value.health.return_value = True
            result = server.start()
        self.assertTrue(result.success)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["server_pid"], current_pid)
        self.assertIsNone(state["worker_pid"])

    def test_active_server_blocks_acquisition_and_stale_server_is_reclaimed(self):
        current_pid = os.getpid()
        current_started = psutil.Process(current_pid).create_time()
        active = {"supervisor_pid": 999999, "supervisor_started_epoch": 1,
                  "server_pid": current_pid, "server_started_epoch": current_started,
                  "worker_pid": None, "worker_started_epoch": None}
        self.state_path.write_text(json.dumps(active), encoding="utf-8")
        with self.assertRaises(RuntimeError): BenchmarkStateStore(self.state_path).acquire()
        active["server_started_epoch"] = current_started + 1000
        self.state_path.write_text(json.dumps(active), encoding="utf-8")
        reclaimed = BenchmarkStateStore(self.state_path)
        reclaimed.acquire()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertIsNone(state["server_pid"])
        reclaimed.release()

    def test_keyboard_interrupt_during_startup_and_model_check_cleans_server(self):
        fake_server = Mock(policy=self.policy, shutdown_confirmed=True)
        fake_server.start.side_effect = KeyboardInterrupt
        with patch("ttea.benchmarking.ollama.OwnedOllamaServer", return_value=fake_server):
            result = self.runtime().run("model", "prompt", ResourceRequest("test", 1, 1))
        self.assertEqual(result.outcome, BenchmarkOutcome.CANCELLED)
        fake_server.shutdown.assert_called_once()
        self.assertFalse(self.state_path.exists())

        fake_client = Mock()
        fake_client.health.return_value = True
        fake_client.installed_models.side_effect = KeyboardInterrupt
        fake_client.running_models.return_value = ()
        fake_server = Mock(policy=self.policy, shutdown_confirmed=True)
        fake_server.start.return_value = OllamaServerStartResult(True, ("healthy",), datetime.now(timezone.utc), datetime.now(timezone.utc), OwnedProcessIdentity(1, 1))
        fake_server.client.return_value = fake_client
        with patch("ttea.benchmarking.ollama.OwnedOllamaServer", return_value=fake_server):
            result = self.runtime().run("model", "prompt", ResourceRequest("test", 1, 1))
        self.assertEqual(result.outcome, BenchmarkOutcome.CANCELLED)
        fake_server.shutdown.assert_called_once()
        self.assertFalse(self.state_path.exists())

    @patch("ttea.benchmarking.ollama.resolve_ollama_executable", return_value="ollama-test")
    def test_identity_capture_failure_stops_exact_spawned_handle(self, resolver):
        class DirectPopen:
            pid = 42
            returncode = None
            def __init__(self): self.alive, self.terminated, self.killed = True, False, False
            def poll(self): return None if self.alive else 0
            def terminate(self): self.terminated, self.alive = True, False
            def kill(self): self.killed, self.alive = True, False
            def wait(self, timeout): return 0
        popen = DirectPopen()
        def unavailable(pid): raise psutil.AccessDenied(pid)
        server = OwnedOllamaServer(self.endpoint, self.policy, process_factory=lambda *args, **kwargs: popen, process_api=unavailable)
        result = server.start()
        self.assertFalse(result.success)
        self.assertTrue(popen.terminated)
        self.assertTrue(server.shutdown_confirmed)

    def test_server_termination_failure_preserves_runtime_state(self):
        current_pid = os.getpid()
        current_started = psutil.Process(current_pid).create_time()
        callback = None
        fake_client = Mock()
        fake_client.health.return_value = True
        fake_client.installed_models.return_value = ()
        fake_client.running_models.return_value = ()
        fake_server = Mock(policy=self.policy, shutdown_confirmed=False)
        def construct(*args, **kwargs):
            nonlocal callback
            callback = kwargs["identity_callback"]
            return fake_server
        def start():
            callback(OwnedProcessIdentity(current_pid, current_started))
            return OllamaServerStartResult(True, ("healthy",), datetime.now(timezone.utc), datetime.now(timezone.utc), OwnedProcessIdentity(current_pid, current_started))
        fake_server.start.side_effect = start
        fake_server.client.return_value = fake_client
        fake_server.shutdown.side_effect = OllamaRuntimeError("still alive")
        with patch("ttea.benchmarking.ollama.OwnedOllamaServer", side_effect=construct):
            result = self.runtime().run("missing", "prompt", ResourceRequest("test", 1, 1))
        self.assertEqual(result.outcome, BenchmarkOutcome.WORKER_FAILURE)
        self.assertTrue(self.state_path.exists())
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["server_pid"], current_pid)

    def test_control_and_generation_http_timeouts_are_distinct(self):
        client = OllamaClient(self.endpoint, self.policy, 2.0)
        with patch("ttea.benchmarking.ollama.http_request.urlopen") as urlopen:
            urlopen.side_effect = [HttpResponse({"version": "1"}), HttpResponse({"model": "m", "response": "r"})]
            client.health()
            client.generate("m", "p")
        self.assertEqual(urlopen.call_args_list[0].kwargs["timeout"], 2.0)
        self.assertEqual(urlopen.call_args_list[1].kwargs["timeout"], self.policy.request_timeout_seconds)

    def test_generation_result_is_pickle_safe(self):
        payload = {"model": "m", "response": "r", "total_duration": 1, "eval_count": 2}
        with patch("ttea.benchmarking.ollama.http_request.urlopen", return_value=HttpResponse(payload)):
            result = self.client.generate("m", "p")
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)

    def test_manual_kill_switch_includes_server_identity_contract(self):
        script = (Path(__file__).parents[1] / "scripts" / "Stop-TTeaBenchmark.ps1").read_text(encoding="utf-8")
        self.assertIn("server_pid", script)
        self.assertIn("server_started_epoch", script)
        self.assertNotIn("-Name ollama", script)

    @patch("ttea.benchmarking.ollama.resolve_ollama_executable", return_value="ollama-test")
    def test_cli_fallback_uses_exact_model_and_server_environment(self, resolver):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        server = OwnedOllamaServer(self.endpoint, self.policy, command_runner=runner)
        server.stop_model_cli("exact:tag", 0.5)
        self.assertEqual(runner.call_args.args[0], ["ollama-test", "stop", "exact:tag"])
        self.assertEqual(runner.call_args.kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11435")
        self.assertEqual(runner.call_args.kwargs["timeout"], 0.5)

    def test_cleanup_skips_cli_after_api_unload_and_always_stops_server(self):
        client, server = Mock(), Mock(policy=self.policy)
        client.running_models.return_value = ()
        cleanup = OllamaCleanup(client, server, "exact").bounded()
        cleanup.run(0.2)
        client.unload.assert_called_once_with("exact")
        server.stop_model_cli.assert_not_called()
        server.shutdown.assert_called_once()

    def test_owned_process_tree_termination_does_not_touch_unrelated_process(self):
        child, unrelated = FakeProcess(2, 20), FakeProcess(3, 30)
        parent = FakeProcess(1, 10, (child,))
        server = OwnedOllamaServer(self.endpoint, self.policy, process_api=lambda pid: parent,
                                   wait_procs=lambda processes, timeout: (processes, []))
        server.identity = OwnedProcessIdentity(1, 10)
        server.shutdown(0.1)
        self.assertTrue(parent.terminated)
        self.assertTrue(child.terminated)
        self.assertFalse(unrelated.terminated)
        self.assertIsNone(server.identity)

    def test_failed_owned_tree_termination_is_surfaced(self):
        parent = FakeProcess(1, 10, survives=True)
        server = OwnedOllamaServer(self.endpoint, self.policy, process_api=lambda pid: parent,
                                   wait_procs=lambda processes, timeout: ([], processes))
        server.identity = OwnedProcessIdentity(1, 10)
        with self.assertRaises(OllamaRuntimeError): server.shutdown(0.01)
        self.assertIsNotNone(server.identity)

    def test_resolver_precedence_and_windows_user_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_executable = root / "path-ollama.exe"
            explicit_executable = root / "explicit-ollama.exe"
            path_executable.touch()
            explicit_executable.touch()
            with patch("ttea.benchmarking.ollama.shutil.which", return_value=str(path_executable)):
                self.assertEqual(resolve_ollama_executable(), str(path_executable))
                self.assertEqual(resolve_ollama_executable(explicit_executable), str(explicit_executable))
            local = root / "local"
            candidate = local / "Programs" / "Ollama" / "ollama.exe"
            candidate.parent.mkdir(parents=True)
            candidate.touch()
            with patch("ttea.benchmarking.ollama.shutil.which", return_value=None), patch("ttea.benchmarking.ollama.os.name", "nt"), patch.dict("ttea.benchmarking.ollama.os.environ", {"LOCALAPPDATA": str(local)}, clear=False):
                self.assertEqual(resolve_ollama_executable(), str(candidate))
            candidate.unlink()
            with patch("ttea.benchmarking.ollama.shutil.which", return_value=None), patch("ttea.benchmarking.ollama.os.name", "nt"), patch.dict("ttea.benchmarking.ollama.os.environ", {"LOCALAPPDATA": str(local)}, clear=False):
                self.assertIsNone(resolve_ollama_executable())

    @patch("ttea.benchmarking.ollama.resolve_ollama_executable", return_value=None)
    def test_unresolved_executable_fails_before_popen(self, resolver):
        popen = Mock()
        result = OwnedOllamaServer(self.endpoint, self.policy, process_factory=popen).start()
        self.assertFalse(result.success)
        self.assertIn("could not be resolved", " ".join(result.reasons))
        popen.assert_not_called()

    def test_resolver_and_passive_check_do_not_mutate_environment(self):
        before = dict(os.environ)
        with patch("ttea.benchmarking.ollama.resolve_ollama_executable", return_value="C:/resolved/ollama.exe"), patch("ttea.benchmarking.ollama.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "ollama version test", "")), patch("ttea.benchmarking.ollama.OllamaClient") as client_type:
            client_type.return_value.version.return_value = "test"
            client_type.return_value.installed_models.return_value = ()
            executable, version, api, models = passive_ollama_check()
        self.assertEqual(os.environ, before)
        self.assertEqual(executable, "C:/resolved/ollama.exe")
        self.assertEqual(version, "ollama version test")
        self.assertEqual(api, "reachable (test)")
        self.assertEqual(models, ())


if __name__ == "__main__":
    unittest.main()
