"""Read-only terminal monitoring for persisted validation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

from ttea.benchmarking import BenchmarkSafetyPolicy, BenchmarkStateStore, OllamaClient, OllamaEndpoint
from ttea.benchmarking.ollama import OllamaRuntimeError, resolve_ollama_executable
from ttea.resources import ResourceManager
from ttea.telemetry import HardwareTelemetryService

from .campaign import CampaignStore, DEFAULT_CAMPAIGN_ROOT, DEFAULT_REPORT_ROOT, DEFAULT_STATE_PATH
from .campaign_models import ACTIVE_CAMPAIGN_STATES, CampaignState, ValidationCampaign
from .model_assets import TINY_MODEL_NAME
from .suites import build_registry, listening_identity, loopback_port_open


MIB = 1024 * 1024
GIB = 1024 * MIB


def human_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= GIB:
        return f"{value / GIB:.2f} GiB"
    return f"{value / MIB:.2f} MiB"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def human_nanoseconds(value: int | float | None) -> str:
    return "-" if value is None else f"{float(value) / 1_000_000_000:.2f} s"


def bounded_tail(path: Path, lines: int = 10, maximum_bytes: int = 16 * 1024) -> tuple[str, ...]:
    """Read only the bounded end of a potentially large text log."""
    if not path.is_file() or lines <= 0 or maximum_bytes <= 0:
        return ()
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            data = handle.read(maximum_bytes)
        return tuple(item for item in data.decode("utf-8", errors="replace").splitlines() if item)[-lines:]
    except OSError:
        return ()


@dataclass(frozen=True)
class CampaignWatchSnapshot:
    campaign: ValidationCampaign | None
    supervisor_alive: bool
    progress_completed: int
    progress_total: int
    scenario_results: tuple[dict[str, Any], ...]
    report: dict[str, Any] | None
    events: tuple[str, ...]
    benchmark_state: dict[str, Any] | None
    telemetry: dict[str, Any] | None
    telemetry_error: str | None
    ollama: dict[str, Any]
    captured_at: datetime

    def to_dict(self) -> dict[str, Any]:
        campaign = self.campaign
        return {
            "campaign": None if campaign is None else {
                "id": campaign.campaign_id, "state": campaign.state.value,
                "target": campaign.target_suite, "current": campaign.current_suite,
                "retries": campaign.retry_count, "reason": campaign.last_reason,
                "next_retry_at": None if campaign.next_retry_at is None else campaign.next_retry_at.isoformat(),
            },
            "supervisor_alive": self.supervisor_alive,
            "progress": {"completed": self.progress_completed, "total": self.progress_total},
            "scenario_results": list(self.scenario_results), "report": self.report,
            "events": list(self.events), "benchmark_state": self.benchmark_state,
            "telemetry": self.telemetry, "telemetry_error": self.telemetry_error,
        "ollama": self.ollama, "captured_at": self.captured_at.isoformat(),
        }


class CampaignWatchReader:
    """Collect a passive, detached snapshot. It never calls mutating campaign APIs."""

    def __init__(self, campaign_root: Path = DEFAULT_CAMPAIGN_ROOT, report_root: Path = DEFAULT_REPORT_ROOT,
                 benchmark_state_path: Path = DEFAULT_STATE_PATH, registry=None, telemetry_factory=None,
                 now: Callable[[], datetime] | None = None) -> None:
        self.store = CampaignStore(campaign_root)
        self.report_root, self.benchmark_state_path = report_root, benchmark_state_path
        self.registry = registry or build_registry()
        self.telemetry_factory = telemetry_factory or (lambda: HardwareTelemetryService(Path.cwd()))
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._ollama_static: tuple[str, str] | None = None

    def _campaign(self) -> ValidationCampaign | None:
        current = self.store.current()
        if current is not None:
            return current
        campaigns = self.store.list()
        return campaigns[-1] if campaigns else None

    def _report(self, campaign: ValidationCampaign | None) -> dict[str, Any] | None:
        if campaign is None or not campaign.history:
            return None
        path = self.report_root / campaign.history[-1].run_id / "report.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            result = dict(value)
            result["watch_paths"] = {
                "campaign_report": str(self.store.campaign_dir(campaign.campaign_id) / "campaign-report.json"),
                "campaign_log": str(self.store.log_path(campaign.campaign_id)), "latest_report": str(path),
            }
            return result
        except (OSError, ValueError):
            return None

    def _all_scenarios(self, campaign: ValidationCampaign | None) -> tuple[dict[str, Any], ...]:
        if campaign is None:
            return ()
        results: dict[tuple[str, str], dict[str, Any]] = {}
        for record in campaign.history:
            try:
                report = json.loads((self.report_root / record.run_id / "report.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for item in report.get("scenario_results", ()) if isinstance(report, dict) else ():
                if isinstance(item, dict) and isinstance(item.get("scenario_id"), str):
                    results[(record.suite, item["scenario_id"])] = dict(item)
        return tuple(results.values())

    def _progress(self, campaign: ValidationCampaign | None, results: tuple[dict[str, Any], ...]) -> tuple[int, int]:
        if campaign is None:
            return 0, 0
        total = sum(len(self.registry.get(name).scenario_factories) for name in campaign.requested_suites)
        completed = sum(1 for item in results if item.get("status") in ("PASS", "FAIL", "SKIP"))
        return min(completed, total), total

    def _telemetry(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            service = self.telemetry_factory()
            observation = service.snapshot()
            budget = ResourceManager(service).current_budget()
            gpu_metrics = {gpu.index: gpu for gpu in observation.gpus}
            return {
                "cpu_percent": observation.cpu.utilization_percent, "ram_available_bytes": budget.available_ram_bytes,
                "ram_reserved_bytes": budget.reserved_ram_bytes, "ram_allocatable_bytes": budget.allocatable_ram_bytes,
                "disk_free_bytes": budget.free_disk_bytes,
                "gpus": [{"index": gpu.index, "name": gpu.name,
                          "free_vram_bytes": gpu.free_vram_bytes, "reserved_vram_bytes": gpu.reserved_vram_bytes,
                          "allocatable_vram_bytes": gpu.allocatable_vram_bytes, "temperature_c": gpu.temperature_c,
                          "max_temperature_c": gpu.max_temperature_c,
                          "utilization_percent": None if gpu.index not in gpu_metrics else gpu_metrics[gpu.index].utilization_percent}
                         for gpu in budget.gpus],
            }, None
        except Exception as error_value:
            return None, str(error_value)

    def _ollama(self, benchmark_state: dict[str, Any] | None) -> dict[str, Any]:
        if self._ollama_static is None:
            executable = resolve_ollama_executable()
            version = "unavailable"
            if executable:
                try:
                    command = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                             timeout=3.0, check=False)
                    if command.returncode == 0:
                        version = (command.stdout or command.stderr).strip() or "unavailable"
                except (OSError, subprocess.SubprocessError):
                    pass
            self._ollama_static = (executable or "not found", version)
        executable, version = self._ollama_static
        default_open, dedicated_open = loopback_port_open(11434), loopback_port_open(11435)
        default_identity = listening_identity(11434) if default_open else None

        def inspect(port: int, open_: bool) -> dict[str, Any]:
            result = {"installed": None, "model_running": False, "api_error": None}
            if not open_:
                return result
            try:
                client = OllamaClient(OllamaEndpoint(f"http://127.0.0.1:{port}"), BenchmarkSafetyPolicy(), 1.0)
                result["installed"] = next((item for item in client.installed_model_records()
                                              if (item.get("name") or item.get("model")) == TINY_MODEL_NAME), None)
                result["model_running"] = TINY_MODEL_NAME in client.running_models()
            except OllamaRuntimeError as error_value:
                result["api_error"] = str(error_value)
            return result

        default, dedicated = inspect(11434, default_open), inspect(11435, dedicated_open)
        locations = [name for name, item in (("DEFAULT", default), ("DEDICATED", dedicated)) if item["model_running"]]
        running_location = "BOTH" if len(locations) == 2 else (locations[0] if locations else "NO")
        owned_pid = None if benchmark_state is None else benchmark_state.get("server_pid")
        return {"executable": executable or "not found", "version": version,
                "default_running": default_open, "default_pid": None if default_identity is None else default_identity.pid,
                "dedicated_running": dedicated_open, "owned_pid": owned_pid,
                "model": TINY_MODEL_NAME, "installed": default["installed"] or dedicated["installed"],
                "model_running": running_location, "default": default, "dedicated": dedicated}

    def read(self) -> CampaignWatchSnapshot:
        campaign = self._campaign()
        alive = bool(campaign and campaign.state in ACTIVE_CAMPAIGN_STATES
                     and self.store.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch))
        report = self._report(campaign)
        scenarios = self._all_scenarios(campaign)
        completed, total = self._progress(campaign, scenarios)
        events = () if campaign is None else bounded_tail(self.store.log_path(campaign.campaign_id))
        state = BenchmarkStateStore(self.benchmark_state_path).snapshot()
        telemetry, telemetry_error = self._telemetry()
        return CampaignWatchSnapshot(campaign, alive, completed, total, scenarios, report, events, state,
                                     telemetry, telemetry_error, self._ollama(state), self.now())


class CampaignWatchRenderer:
    def __init__(self, color: bool = True, width: int | None = None) -> None:
        self.color = color and sys.stdout.isatty()
        self.width = width or shutil.get_terminal_size((100, 30)).columns

    def _style(self, text: str, status: str) -> str:
        if not self.color:
            return text
        codes = {"PASS": "32", "COMPLETED": "32", "RUNNING": "36", "WAITING_FOR_RESOURCES": "33",
                 "BLOCKED": "33", "FAIL": "31", "FAILED": "31", "INTERRUPTED": "33", "CANCELLED": "33"}
        return f"\x1b[{codes[status]}m{text}\x1b[0m" if status in codes else text

    def _line(self, label: str, value: Any) -> str:
        space = max(8, min(14, self.width // 5))
        text = f"{label:<{space}} {value}"
        return text if len(text) <= self.width else text[:max(0, self.width - 3)] + "..."

    @staticmethod
    def _countdown(campaign: ValidationCampaign, now: datetime) -> str:
        if campaign.next_retry_at is None:
            return "-"
        return human_duration((campaign.next_retry_at - now).total_seconds())

    def render(self, snapshot: CampaignWatchSnapshot) -> str:
        if snapshot.campaign is None:
            return "T/TEA CAMPAIGN MONITOR\n\nNo validation campaign or report is available.\n\nCtrl+C exits monitor only.\n"
        campaign = snapshot.campaign
        elapsed_end = snapshot.captured_at if campaign.state in ACTIVE_CAMPAIGN_STATES else campaign.updated_at
        elapsed = (elapsed_end - campaign.created_at).total_seconds()
        total = max(1, snapshot.progress_total)
        width = max(8, min(24, self.width - 28))
        filled = round(width * snapshot.progress_completed / total)
        bar = "[" + "#" * filled + "-" * (width - filled) + "]"
        lines = ["T/TEA CAMPAIGN MONITOR", "=" * min(self.width, 52), "", "CAMPAIGN",
                 self._line("ID", campaign.campaign_id),
                 self._style(self._line("State", campaign.state.value), campaign.state.value),
                 self._line("Target", campaign.target_suite), self._line("Current", campaign.current_suite or "-"),
                 self._line("Supervisor", "RUNNING" if snapshot.supervisor_alive else "STOPPED"),
                 self._line("Elapsed", human_duration(elapsed)), self._line("Retries", campaign.retry_count), "",
                 "PROGRESS", self._line(bar, f"{snapshot.progress_completed} / {snapshot.progress_total}"),
                 self._line("Percent", f"{(100 * snapshot.progress_completed / total):.0f}%")]
        if campaign.history:
            latest = campaign.history[-1]
            lines.append(self._line("Latest result", f"{latest.status.value} {latest.suite}"))
        if campaign.last_reason:
            lines.append(self._line("Reason", campaign.last_reason))
        if campaign.state is CampaignState.RUNNING and snapshot.benchmark_state:
            lines.append(self._line("Worker PID", snapshot.benchmark_state.get("worker_pid") or "-"))
        if campaign.state is CampaignState.WAITING_FOR_RESOURCES:
            lines.extend(("", self._style("WAITING FOR RESOURCES", "WAITING_FOR_RESOURCES"),
                          self._line("Reason", campaign.last_reason or "resource preflight pending"),
                          self._line("Next retry", self._countdown(campaign, snapshot.captured_at))))
            if snapshot.telemetry is not None:
                shortfall = max(0, snapshot.telemetry["ram_reserved_bytes"] - snapshot.telemetry["ram_available_bytes"])
                lines.append(self._line("RAM shortfall", human_bytes(shortfall)))
        lines.extend(("", "SYSTEM"))
        if snapshot.telemetry is None:
            lines.append(self._line("Telemetry unavailable", snapshot.telemetry_error or "unknown reason"))
        else:
            item = snapshot.telemetry
            lines.extend((self._line("CPU", f"{item['cpu_percent']} %"),
                          self._line("RAM", f"{human_bytes(item['ram_available_bytes'])} available"),
                          self._line("Reserve", human_bytes(item['ram_reserved_bytes'])),
                          self._line("Allocatable", human_bytes(item['ram_allocatable_bytes'])),
                          self._line("Disk free", human_bytes(item['disk_free_bytes']))))
            for gpu in item["gpus"]:
                lines.extend((self._line(f"GPU {gpu['index']}", gpu["name"]),
                              self._line("  Utilization", "-" if gpu["utilization_percent"] is None else f"{gpu['utilization_percent']} %"),
                              self._line("  VRAM free", human_bytes(gpu["free_vram_bytes"])),
                              self._line("  Reserve", human_bytes(gpu["reserved_vram_bytes"])),
                              self._line("  Allocatable", human_bytes(gpu["allocatable_vram_bytes"])),
                              self._line("  Temperature", f"{gpu['temperature_c'] if gpu['temperature_c'] is not None else '-'} C / {gpu['max_temperature_c']} C")))
        ollama = snapshot.ollama
        lines.extend(("", "OLLAMA", self._line("Executable", ollama["executable"]), self._line("Version", ollama["version"]),
                      self._line("Default", "RUNNING 127.0.0.1:11434" if ollama["default_running"] else "STOPPED 127.0.0.1:11434"),
                      self._line("Default PID", ollama["default_pid"] or "-"),
                      self._line("Dedicated", "RUNNING 127.0.0.1:11435" if ollama["dedicated_running"] else "STOPPED 127.0.0.1:11435"),
                      self._line("Owned PID", ollama["owned_pid"] or "-"), self._line("Model", ollama["model"]),
                      self._line("Installed", "YES" if ollama["installed"] else "NO"),
                      self._line("Running", ollama["model_running"])))
        summary = dict((snapshot.report or {}).get("safety_summary", {}) or {})
        inference = next((item.get("evidence", {}) for item in (snapshot.report or {}).get("scenario_results", ())
                          if isinstance(item, dict) and item.get("scenario_id") == "tiny_model_inference"), {})
        if isinstance(inference, dict):
            summary = {**inference, **summary}
        if isinstance(summary, dict):
            lines.extend(("", "LATEST METRICS"))
            for label, key in (("Model", "model"), ("Parameters", "parameter_size"), ("Quantization", "quantization"),
                               ("Decode", "tokens_per_second"), ("Tokens", "eval_count"), ("Peak GPU temp", "gpu_max_temperature_c"),
                               ("Digest", "digest"), ("Model size", "size_bytes"), ("RAM delta", "ram_delta_bytes"),
                               ("GPU delta", "gpu_delta_bytes"), ("Unload", "model_unload_confirmed"),
                               ("Cleanup", "server_cleanup_confirmed"), ("Timeout cleanup", "timeout_cleanup_verified"),
                               ("Default preserved", "default_server_identity_preserved"), ("Resources", "resources_recovered"),
                               ("Timeout", "forced_timeout_outcome")):
                if key in summary:
                    lines.append(self._line(label, summary[key]))
            for label, key in (("Total duration", "total_duration"), ("Load duration", "load_duration"),
                               ("Prompt eval", "prompt_eval_duration"), ("Eval duration", "eval_duration")):
                if key in summary:
                    lines.append(self._line(label, human_nanoseconds(summary[key])))
        paths = (snapshot.report or {}).get("watch_paths", {})
        if isinstance(paths, dict):
            lines.extend(("", "REPORTS", self._line("Campaign", paths.get("campaign_report", "-")),
                          self._line("Latest", paths.get("latest_report", "-")), self._line("Log", paths.get("campaign_log", "-"))))
        if snapshot.scenario_results:
            lines.extend(("", "VALIDATION"))
            lines.extend(self._line(str(item.get("status", "?")), item.get("scenario_id", "?")) for item in snapshot.scenario_results)
        lines.extend(("", "RECENT EVENTS"))
        lines.extend(item if len(item) <= self.width else item[:max(0, self.width - 3)] + "..."
                     for item in (snapshot.events or ("(none)",)))
        lines.extend(("", "Ctrl+C exits monitor only."))
        return "\n".join(lines) + "\n"


class CampaignWatcher:
    def __init__(self, reader: CampaignWatchReader | None = None, renderer: CampaignWatchRenderer | None = None,
                 refresh_seconds: float = 2.0, output=None, sleep: Callable[[float], None] = time.sleep,
                 redraw_supported: bool | None = None) -> None:
        if refresh_seconds < 0.5:
            raise ValueError("refresh_seconds must be at least 0.5")
        self.reader, self.renderer, self.refresh_seconds = reader or CampaignWatchReader(), renderer or CampaignWatchRenderer(), refresh_seconds
        self.output, self.sleep = output or sys.stdout, sleep
        self.redraw_supported = self.output.isatty() if redraw_supported is None else redraw_supported

    def run(self, *, once: bool = False, json_output: bool = False) -> int:
        try:
            while True:
                snapshot = self.reader.read()
                text = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n" if json_output else self.renderer.render(snapshot)
                if not once and not json_output and self.redraw_supported:
                    self.output.write("\x1b[2J\x1b[H")
                self.output.write(text)
                self.output.flush()
                if once or json_output:
                    return 0
                self.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            return 0
