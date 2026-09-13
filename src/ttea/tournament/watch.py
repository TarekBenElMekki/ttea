"""Read-only native terminal monitor for survival tournaments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, TextIO

from ttea.benchmarking import BenchmarkStateStore
from ttea.resources import ResourceManager
from ttea.telemetry import HardwareTelemetryService

from .execution import ACTIVE_SURVIVAL_STATES, SurvivalCampaign, SurvivalOutcome, SurvivalStore


def _tail(path: Path, limit: int = 12) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    return tuple(lines[-limit:])


@dataclass(frozen=True)
class TournamentWatchSnapshot:
    campaign: SurvivalCampaign | None
    supervisor_alive: bool
    captured_at: datetime
    benchmark_state: dict[str, Any] | None
    system: dict[str, Any] | None
    system_error: str | None
    events: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        campaign = self.campaign
        return {
            "tournament": None if campaign is None else {
                "id": campaign.tournament_id, "state": campaign.state.value,
                "candidate": campaign.current_candidate_id, "round": campaign.current_round_id,
                "progress": list(campaign.progress), "reason": campaign.last_reason,
                "results": {key: value.outcome.value for key, value in campaign.records.items()},
            },
            "supervisor_alive": self.supervisor_alive,
            "captured_at": self.captured_at.isoformat(), "benchmark_state": self.benchmark_state,
            "system": self.system, "system_error": self.system_error, "events": list(self.events),
        }


class TournamentWatcher:
    """Collect passive snapshots without calling any mutating tournament API."""

    def __init__(self, root: Path = Path("data/tournaments"),
                 benchmark_state_path: Path = Path("data/runtime/benchmark.json"),
                 resource_manager: ResourceManager | None = None) -> None:
        self.store = SurvivalStore(root)
        self.benchmark_store = BenchmarkStateStore(benchmark_state_path)
        self.resource_manager = resource_manager or ResourceManager(HardwareTelemetryService(Path.cwd()))

    def snapshot(self) -> TournamentWatchSnapshot:
        campaign = self.store.latest()
        alive = bool(campaign and campaign.state in ACTIVE_SURVIVAL_STATES and
                     BenchmarkStateStore.process_matches(campaign.supervisor_pid,
                                                         campaign.supervisor_started_epoch))
        benchmark = self.benchmark_store.snapshot()
        system = None
        error = None
        try:
            budget = self.resource_manager.current_budget()
            system = {
                "ram_available_bytes": budget.available_ram_bytes,
                "ram_reserved_bytes": budget.reserved_ram_bytes,
                "ram_allocatable_bytes": budget.allocatable_ram_bytes,
                "gpus": [{
                    "index": gpu.index, "name": gpu.name,
                    "free_vram_bytes": gpu.free_vram_bytes,
                    "allocatable_vram_bytes": gpu.allocatable_vram_bytes,
                    "temperature_c": gpu.temperature_c,
                } for gpu in budget.gpus],
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        events = () if campaign is None else _tail(self.store.tournaments.events_path(campaign.tournament_id))
        return TournamentWatchSnapshot(campaign, alive, datetime.now(timezone.utc), benchmark, system, error, events)


class TournamentWatchRenderer:
    def render(self, snapshot: TournamentWatchSnapshot) -> str:
        if snapshot.campaign is None:
            return "T/TEA TOURNAMENT MONITOR\n\nNo survival tournament is available.\n\nCtrl+C exits viewer only.\n"
        campaign = snapshot.campaign
        completed, total = campaign.progress
        current = campaign.records.get(campaign.current_candidate_id or "")
        lines = [
            "T/TEA TOURNAMENT MONITOR", "", "TOURNAMENT",
            f"ID: {campaign.tournament_id}", f"State: {campaign.state.value}",
            "Phase: SURVIVAL", f"Candidate: {campaign.current_candidate_id or '-'}",
            f"Round/task: {campaign.current_round_id or '-'}/survival_probe",
            f"Elapsed: {max(0.0, (snapshot.captured_at - campaign.created_at).total_seconds()):.1f}s",
            f"Progress: {completed}/{total}", "", "CURRENT CANDIDATE",
            f"Tag: {current.model if current else '-'}",
            f"Provisioning state: {current.provisioning_state.value if current else '-'}",
            f"Attempt: {current.attempt_count if current else '-'}",
            f"Worker PID: {(snapshot.benchmark_state or {}).get('worker_pid') or '-'}",
            f"Owned Ollama PID: {(snapshot.benchmark_state or {}).get('server_pid') or '-'}",
            "", "SYSTEM",
        ]
        if snapshot.system is None:
            lines.append(f"Telemetry unavailable: {snapshot.system_error or 'unknown'}")
        else:
            system = snapshot.system
            lines.extend((
                "CPU: passive telemetry snapshot",
                f"RAM available/reserve/allocatable: {system['ram_available_bytes']}/{system['ram_reserved_bytes']}/{system['ram_allocatable_bytes']}",
            ))
            if not system["gpus"]:
                lines.append("GPU: unavailable")
            for gpu in system["gpus"]:
                lines.append(f"GPU {gpu['index']} {gpu['name']}: VRAM free/allocatable {gpu['free_vram_bytes']}/{gpu['allocatable_vram_bytes']}; temperature {gpu['temperature_c']}")
        lines.extend(("", "ROUND 1 RESULTS"))
        labels = {
            SurvivalOutcome.PENDING: "PENDING", SurvivalOutcome.WAITING_FOR_RESOURCES: "WAITING",
            SurvivalOutcome.RUNNING: "RUNNING", SurvivalOutcome.PASSED: "PASS",
            SurvivalOutcome.ELIMINATED: "ELIMINATED", SurvivalOutcome.BLOCKED: "BLOCKED",
            SurvivalOutcome.INTERRUPTED: "PENDING", SurvivalOutcome.CANCELLED: "CANCELLED",
        }
        for record in campaign.records.values():
            state = "DOWNLOADING" if record.provisioning_state.value == "DOWNLOADING" else labels[record.outcome]
            lines.append(f"{record.candidate_id:<25} {state}")
        lines.extend(("", "RECENT EVENTS"))
        lines.extend(snapshot.events or ("(none)",))
        lines.extend(("", "Ctrl+C exits viewer only."))
        return "\n".join(lines) + "\n"


def watch(root: Path, *, refresh_seconds: float = 2.0, once: bool = False,
          json_output: bool = False, output: TextIO = sys.stdout,
          sleep: Callable[[float], None] = time.sleep,
          watcher: TournamentWatcher | None = None) -> int:
    if refresh_seconds < 0.5:
        raise ValueError("refresh must be at least 0.5 seconds")
    watcher = watcher or TournamentWatcher(root)
    renderer = TournamentWatchRenderer()
    try:
        while True:
            snapshot = watcher.snapshot()
            output.write((json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n")
                         if json_output else renderer.render(snapshot))
            output.flush()
            if once or json_output:
                return 0
            sleep(refresh_seconds)
    except KeyboardInterrupt:
        return 0

