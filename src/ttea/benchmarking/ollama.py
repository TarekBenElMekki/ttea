"""Controlled, loopback-only Ollama plumbing for supervised benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from urllib import error, parse, request as http_request

import psutil

from ttea.resources import ResourceDecision, ResourceManager, ResourceRequest

from .models import BenchmarkOutcome, BenchmarkResult, BenchmarkSafetyPolicy
from .safety import BenchmarkStateStore, IdempotentCleanup
from .supervisor import BenchmarkSupervisor, BenchmarkWorker


class OllamaRuntimeError(RuntimeError):
    pass


def _existing_file(candidate: str | Path | None) -> str | None:
    if candidate is None:
        return None
    path = Path(candidate)
    return str(path) if path.is_file() else None


def resolve_ollama_executable(executable: str | Path | None = None) -> str | None:
    """Resolve only an explicit executable, PATH, or the Windows user install path."""
    if executable is not None:
        explicit = str(executable).strip()
        if not explicit:
            return None
        direct = _existing_file(explicit)
        if direct is not None:
            return direct
        return _existing_file(shutil.which(explicit))
    from_path = _existing_file(shutil.which("ollama"))
    if from_path is not None:
        return from_path
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return _existing_file(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
    return None


def _positive_timeout(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class OllamaEndpoint:
    base_url: str = "http://127.0.0.1:11435"

    def __post_init__(self) -> None:
        parsed = parse.urlsplit(self.base_url)
        if parsed.scheme != "http" or parsed.username is not None or parsed.password is not None or parsed.path.rstrip("/") or parsed.query or parsed.fragment:
            raise ValueError("Ollama endpoint must be a plain loopback HTTP origin")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Ollama endpoint must have a valid port") from exc
        if parsed.hostname is None or port is None:
            raise ValueError("Ollama endpoint must include a loopback host and port")
        host = parsed.hostname
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback:
            raise ValueError("Ollama endpoint must be loopback-only")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def host_port(self) -> str:
        parsed = parse.urlsplit(self.base_url)
        host = f"[{parsed.hostname}]" if ":" in (parsed.hostname or "") else parsed.hostname
        return f"{host}:{parsed.port}"


@dataclass(frozen=True)
class OllamaGenerationResult:
    model: str
    response: str
    total_duration: int | None
    load_duration: int | None
    prompt_eval_count: int | None
    prompt_eval_duration: int | None
    eval_count: int | None
    eval_duration: int | None
    done_reason: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OllamaGenerationResult":
        model, response = payload.get("model"), payload.get("response")
        if not isinstance(model, str) or not model.strip() or not isinstance(response, str):
            raise OllamaRuntimeError("Ollama generation response is missing required fields")
        numeric_names = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")
        values: dict[str, int | None] = {}
        for name in numeric_names:
            value = payload.get(name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise OllamaRuntimeError(f"Ollama generation response has invalid {name}")
            values[name] = value
        done_reason = payload.get("done_reason")
        if done_reason is not None and not isinstance(done_reason, str):
            raise OllamaRuntimeError("Ollama generation response has invalid done_reason")
        return cls(model, response, **values, done_reason=done_reason)


class OllamaClient:
    """Minimal standard-library client with an explicit timeout on every call."""

    def __init__(self, endpoint: OllamaEndpoint, policy: BenchmarkSafetyPolicy, control_timeout_seconds: float) -> None:
        _positive_timeout(control_timeout_seconds, "control_timeout_seconds")
        self.endpoint, self.policy, self.control_timeout_seconds = endpoint, policy, control_timeout_seconds

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout_seconds: float | None = None) -> dict[str, Any]:
        timeout = self.control_timeout_seconds if timeout_seconds is None else timeout_seconds
        _positive_timeout(timeout, "timeout_seconds")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if data is None else {"Content-Type": "application/json"}
        outgoing = http_request.Request(self.endpoint.base_url + path, data=data, headers=headers, method=method)
        try:
            with http_request.urlopen(outgoing, timeout=timeout) as response:
                body = response.read()
        except (error.URLError, TimeoutError, OSError) as exc:
            raise OllamaRuntimeError(f"Ollama HTTP request failed: {exc}") from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise OllamaRuntimeError("Ollama returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaRuntimeError("Ollama returned a non-object response")
        return decoded

    def version(self, timeout_seconds: float | None = None) -> str:
        value = self._request("GET", "/api/version", timeout_seconds=timeout_seconds).get("version")
        if not isinstance(value, str) or not value.strip():
            raise OllamaRuntimeError("Ollama version response is invalid")
        return value

    def health(self, timeout_seconds: float | None = None) -> bool:
        self.version(timeout_seconds)
        return True

    @staticmethod
    def _model_names(payload: dict[str, Any], key: str) -> tuple[str, ...]:
        models = payload.get(key)
        if not isinstance(models, list):
            raise OllamaRuntimeError("Ollama model-list response is invalid")
        names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str) and name.strip():
                    names.append(name)
        return tuple(names)

    def installed_models(self) -> tuple[str, ...]:
        return self._model_names(self._request("GET", "/api/tags"), "models")

    def running_models(self) -> tuple[str, ...]:
        return self._model_names(self._request("GET", "/api/ps"), "models")

    def generate(self, model: str, prompt: str, *, num_ctx: int | None = None,
                 num_predict: int | None = None, num_thread: int | None = None) -> OllamaGenerationResult:
        if not isinstance(model, str) or not model.strip() or not isinstance(prompt, str) or not prompt:
            raise ValueError("model and prompt must be non-empty strings")
        bounds = {
            "num_ctx": self.policy.max_context_tokens if num_ctx is None else num_ctx,
            "num_predict": self.policy.max_output_tokens if num_predict is None else num_predict,
            "num_thread": self.policy.cpu_thread_limit if num_thread is None else num_thread,
        }
        maxima = {"num_ctx": self.policy.max_context_tokens, "num_predict": self.policy.max_output_tokens, "num_thread": self.policy.cpu_thread_limit}
        for name, value in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > maxima[name]:
                raise ValueError(f"{name} must be a positive integer no greater than its safety-policy bound")
        payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0, "options": bounds}
        return OllamaGenerationResult.from_payload(
            self._request("POST", "/api/generate", payload, self.policy.request_timeout_seconds)
        )

    def unload(self, model: str) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self._request("POST", "/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0})


@dataclass(frozen=True)
class OwnedProcessIdentity:
    pid: int
    started_epoch: float


@dataclass(frozen=True)
class OllamaServerStartResult:
    success: bool
    reasons: tuple[str, ...]
    started_at: datetime
    finished_at: datetime
    identity: OwnedProcessIdentity | None = None


class OwnedOllamaServer:
    """A dedicated Ollama server whose exact process tree is owned by T/TEA."""

    def __init__(self, endpoint: OllamaEndpoint, policy: BenchmarkSafetyPolicy, *, executable: str | Path | None = None,
                 startup_timeout_seconds: float = 15.0, http_timeout_seconds: float = 2.0,
                 process_factory=None, process_api=None, wait_procs=None, command_runner=None,
                 identity_callback=None) -> None:
        _positive_timeout(startup_timeout_seconds, "startup_timeout_seconds")
        _positive_timeout(http_timeout_seconds, "http_timeout_seconds")
        self.endpoint, self.policy, self.executable = endpoint, policy, executable
        self.startup_timeout_seconds, self.http_timeout_seconds = startup_timeout_seconds, http_timeout_seconds
        self._process_factory = process_factory or subprocess.Popen
        self._process_api = process_api or psutil.Process
        self._wait_procs = wait_procs or psutil.wait_procs
        self._command_runner = command_runner or subprocess.run
        self._identity_callback = identity_callback
        self.process = None
        self.identity: OwnedProcessIdentity | None = None

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({"OLLAMA_HOST": self.endpoint.host_port, "OLLAMA_MAX_LOADED_MODELS": "1",
                            "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_QUEUE": "1",
                            "OLLAMA_CONTEXT_LENGTH": str(self.policy.max_context_tokens),
                            "OLLAMA_KEEP_ALIVE": "0", "OLLAMA_NO_CLOUD": "1"})
        return environment

    def client(self) -> OllamaClient:
        return OllamaClient(self.endpoint, self.policy, self.http_timeout_seconds)

    def start(self) -> OllamaServerStartResult:
        started_at = datetime.now(timezone.utc)
        if self.identity is not None:
            return OllamaServerStartResult(False, ("owned Ollama server is already started",), started_at, datetime.now(timezone.utc), self.identity)
        resolved_executable = resolve_ollama_executable(self.executable)
        if resolved_executable is None:
            return OllamaServerStartResult(False, ("Ollama executable could not be resolved; server was not started",), started_at, datetime.now(timezone.utc))
        kwargs: dict[str, Any] = {"env": self.environment(), "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            self.process = self._process_factory([resolved_executable, "serve"], **kwargs)
            created = self._process_api(self.process.pid).create_time()
            self.identity = OwnedProcessIdentity(self.process.pid, created)
            if self._identity_callback is not None:
                self._identity_callback(self.identity)
            deadline = time.monotonic() + self.startup_timeout_seconds
            client = self.client()
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise OllamaRuntimeError(f"owned Ollama server exited with code {self.process.returncode}")
                try:
                    client.health(min(self.http_timeout_seconds, max(0.001, deadline - time.monotonic())))
                    return OllamaServerStartResult(True, ("owned Ollama server is healthy",), started_at, datetime.now(timezone.utc), self.identity)
                except OllamaRuntimeError:
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            raise OllamaRuntimeError("owned Ollama server startup timed out")
        except BaseException as error_value:
            cleanup_error = None
            try:
                self.shutdown(self.policy.cooldown_timeout_seconds)
            except BaseException as shutdown_error:
                cleanup_error = shutdown_error
            if isinstance(error_value, (KeyboardInterrupt, SystemExit)):
                raise
            reasons = (f"owned Ollama server startup failed: {error_value}",)
            if cleanup_error is not None:
                reasons += (f"owned server cleanup failed: {cleanup_error}",)
            return OllamaServerStartResult(False, reasons, started_at, datetime.now(timezone.utc), self.identity)

    @staticmethod
    def _matches(process, identity: OwnedProcessIdentity) -> bool:
        try:
            return process.pid == identity.pid and abs(process.create_time() - identity.started_epoch) <= 2.0
        except (psutil.Error, OSError, ValueError):
            return False

    def shutdown(self, timeout_seconds: float) -> None:
        _positive_timeout(timeout_seconds, "timeout_seconds")
        if self.identity is None:
            self._shutdown_spawn_handle(timeout_seconds)
            return
        identity = self.identity
        try:
            parent = self._process_api(identity.pid)
        except (psutil.Error, OSError):
            self._shutdown_spawn_handle(timeout_seconds)
            return
        if not self._matches(parent, identity):
            self._shutdown_spawn_handle(timeout_seconds)
            return
        descendants = []
        try:
            for child in parent.children(recursive=True):
                child_identity = OwnedProcessIdentity(child.pid, child.create_time())
                if self._matches(child, child_identity):
                    descendants.append((child, child_identity))
        except (psutil.Error, OSError) as exc:
            raise OllamaRuntimeError(f"could not enumerate owned Ollama process tree: {exc}") from exc
        targets = descendants + [(parent, identity)]
        for process, owned_identity in targets:
            if self._matches(process, owned_identity):
                process.terminate()
        processes = [process for process, _ in targets]
        phase_timeout = timeout_seconds / 2
        _, alive = self._wait_procs(processes, timeout=phase_timeout)
        for process in alive:
            matching = next((owned for candidate, owned in targets if candidate.pid == process.pid), None)
            if matching is not None and self._matches(process, matching):
                process.kill()
        if alive:
            self._wait_procs(alive, timeout=phase_timeout)
        still_owned = []
        for process, owned_identity in targets:
            if self._matches(process, owned_identity) and process.is_running():
                still_owned.append(process.pid)
        if still_owned:
            raise OllamaRuntimeError(f"owned Ollama processes remain alive: {still_owned}")
        self.identity = None
        self.process = None

    def _shutdown_spawn_handle(self, timeout_seconds: float) -> None:
        """Boundedly stop the exact Popen handle when strong identity capture is unavailable."""
        if self.process is None:
            return
        phase_timeout = timeout_seconds / 2
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=phase_timeout)
                except subprocess.TimeoutExpired:
                    pass
            if self.process.poll() is None:
                self.process.kill()
                try:
                    self.process.wait(timeout=phase_timeout)
                except subprocess.TimeoutExpired:
                    pass
            if self.process.poll() is None:
                raise OllamaRuntimeError(f"exact spawned Ollama PID {self.process.pid} remains alive")
        except OllamaRuntimeError:
            raise
        except Exception as error_value:
            raise OllamaRuntimeError(f"exact spawned Ollama PID {self.process.pid} shutdown failed: {error_value}") from error_value
        self.identity = None
        self.process = None

    @property
    def shutdown_confirmed(self) -> bool:
        if self.identity is not None:
            return False
        if self.process is None:
            return True
        try:
            return self.process.poll() is not None
        except Exception:
            return False

    def stop_model_cli(self, model: str, timeout_seconds: float) -> None:
        _positive_timeout(timeout_seconds, "timeout_seconds")
        resolved_executable = resolve_ollama_executable(self.executable)
        if resolved_executable is None:
            raise OllamaRuntimeError("Ollama executable could not be resolved for exact-model CLI cleanup")
        result = self._command_runner([resolved_executable, "stop", model], env=self.environment(), capture_output=True,
                                      text=True, timeout=timeout_seconds, check=False)
        if result.returncode != 0:
            raise OllamaRuntimeError(f"exact-model CLI stop failed with code {result.returncode}")


class OllamaCleanup:
    def __init__(self, client: OllamaClient, server: OwnedOllamaServer, model: str) -> None:
        self.client, self.server, self.model = client, server, model
        self._api_unloaded = False

    def api_unload(self) -> None:
        self.client.unload(self.model)
        self._api_unloaded = self.model not in self.client.running_models()
        if not self._api_unloaded:
            raise OllamaRuntimeError("model remained loaded after API unload")

    def cli_stop(self) -> None:
        if not self._api_unloaded:
            self.server.stop_model_cli(self.model, self.server.policy.cooldown_timeout_seconds)

    def shutdown_server(self) -> None:
        self.server.shutdown(self.server.policy.cooldown_timeout_seconds)

    def bounded(self) -> IdempotentCleanup:
        return IdempotentCleanup((self.api_unload, self.cli_stop, self.shutdown_server))


class OllamaBenchmarkWorker(BenchmarkWorker):
    """Exactly one local model, prompt, and bounded non-streaming generation."""

    def __init__(self, client: OllamaClient, model: str, prompt: str) -> None:
        if not isinstance(model, str) or not model.strip() or not isinstance(prompt, str) or not prompt:
            raise ValueError("model and prompt must be non-empty strings")
        self.client, self.model, self.prompt = client, model, prompt

    def run(self) -> OllamaGenerationResult:
        return self.client.generate(self.model, self.prompt)


class ControlledOllamaRuntime:
    """Canonical ResourceManager -> BenchmarkSupervisor -> Ollama worker path."""

    def __init__(self, resource_manager: ResourceManager, policy: BenchmarkSafetyPolicy | None = None,
                 endpoint: OllamaEndpoint | None = None, *, executable: str | Path | None = None,
                 startup_timeout_seconds: float = 15.0, control_timeout_seconds: float = 2.0,
                 state_path: Path = Path("data/runtime/benchmark.json")) -> None:
        _positive_timeout(control_timeout_seconds, "control_timeout_seconds")
        self.resource_manager = resource_manager
        self.policy = policy or BenchmarkSafetyPolicy()
        self.endpoint = endpoint or OllamaEndpoint()
        self.executable, self.startup_timeout_seconds = executable, startup_timeout_seconds
        self.control_timeout_seconds, self.state_path = control_timeout_seconds, state_path

    @staticmethod
    def _failure(started: datetime, *reasons: str) -> BenchmarkResult:
        return BenchmarkResult(BenchmarkOutcome.WORKER_FAILURE, tuple(reasons), started, datetime.now(timezone.utc))

    def run(self, model: str, prompt: str, resource_request: ResourceRequest) -> BenchmarkResult:
        started = datetime.now(timezone.utc)
        try:
            assessment = self.resource_manager.assess(resource_request)
        except KeyboardInterrupt:
            return BenchmarkResult(BenchmarkOutcome.CANCELLED, ("benchmark cancelled during resource preflight",), started, datetime.now(timezone.utc))
        if assessment.decision is not ResourceDecision.ALLOW:
            return BenchmarkResult(BenchmarkOutcome.RESOURCE_PRESSURE, assessment.reasons, started, datetime.now(timezone.utc))

        state_store = BenchmarkStateStore(self.state_path)
        try:
            state_store.acquire()
        except KeyboardInterrupt:
            return BenchmarkResult(BenchmarkOutcome.CANCELLED, ("benchmark cancelled before server startup",), started, datetime.now(timezone.utc))
        except Exception as error_value:
            return self._failure(started, f"benchmark ownership acquisition failed: {error_value}")

        server = OwnedOllamaServer(
            self.endpoint, self.policy, executable=self.executable,
            startup_timeout_seconds=self.startup_timeout_seconds,
            http_timeout_seconds=self.control_timeout_seconds,
            identity_callback=lambda identity: state_store.record_server(identity.pid, identity.started_epoch),
        )
        cleanup = None
        try:
            startup = server.start()
            if not startup.success:
                if server.shutdown_confirmed:
                    state_store.release()
                return self._failure(started, *startup.reasons)
            client = server.client()
            cleanup = OllamaCleanup(client, server, model).bounded()
            client.health()
            if model not in client.installed_models():
                raise OllamaRuntimeError(f"model {model!r} is not installed locally; automatic pulls are forbidden")
            supervisor = BenchmarkSupervisor(
                self.resource_manager, self.policy, cleanup, self.state_path,
                state_store=state_store, state_already_acquired=True,
            )
            return supervisor.run(OllamaBenchmarkWorker(client, model, prompt), resource_request)
        except KeyboardInterrupt:
            outcome = BenchmarkOutcome.CANCELLED
            reasons = ["benchmark cancelled before supervised generation"]
        except Exception as error_value:
            outcome = BenchmarkOutcome.WORKER_FAILURE
            reasons = [f"Ollama benchmark setup failed: {error_value}"]
        except BaseException:
            self._cleanup_before_supervision(cleanup, server, state_store)
            raise
        cleanup_error = self._cleanup_before_supervision(cleanup, server, state_store)
        if cleanup_error is not None:
            reasons.append(f"cleanup failed: {cleanup_error}")
        return BenchmarkResult(outcome, tuple(reasons), started, datetime.now(timezone.utc))

    def _cleanup_before_supervision(self, cleanup: IdempotentCleanup | None,
                                    server: OwnedOllamaServer,
                                    state_store: BenchmarkStateStore) -> str | None:
        cleanup_error = None
        try:
            if cleanup is None:
                server.shutdown(self.policy.cooldown_timeout_seconds)
            else:
                cleanup.run(self.policy.cooldown_timeout_seconds)
        except BaseException as error_value:
            cleanup_error = str(error_value)
        if server.shutdown_confirmed:
            try:
                state_store.release()
            except Exception as error_value:
                cleanup_error = cleanup_error or str(error_value)
        elif cleanup_error is None:
            cleanup_error = "owned Ollama server shutdown could not be confirmed; ownership state was preserved"
        return cleanup_error


def passive_ollama_check() -> tuple[str, str, str, tuple[str, ...] | None]:
    """Discover executable/default local API status without launching or generating."""
    executable = resolve_ollama_executable()
    executable_text = executable or "not found"
    version = "unavailable"
    if executable:
        try:
            result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=3.0, check=False)
            if result.returncode == 0:
                version = (result.stdout or result.stderr).strip() or "unavailable"
        except (OSError, subprocess.SubprocessError):
            pass
    client = OllamaClient(OllamaEndpoint("http://127.0.0.1:11434"), BenchmarkSafetyPolicy(), 1.0)
    try:
        api_version = client.version()
        models = client.installed_models()
        api = f"reachable ({api_version})"
    except OllamaRuntimeError:
        api, models = "not reachable", None
    return executable_text, version, api, models if api.startswith("reachable") else None
