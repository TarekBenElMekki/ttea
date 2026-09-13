"""Read-only checklist-style terminal monitor for survival tournaments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

import psutil

from ttea.resources import ResourceDecision

from .ollama_provider import survival_resource_request
from .watch import TournamentWatchSnapshot, TournamentWatcher


_GIB = 1024 * 1024 * 1024

# RSS is not equivalent to immediately reclaimable Windows memory. Use a
# deliberately conservative estimate for the "likely" projection.
_CHROME_CONSERVATIVE_RECLAIM_RATIO = 0.60

_CHROME_CLOSED_RSS_THRESHOLD_BYTES = 64 * 1024 * 1024
_CHROME_SIGNIFICANT_RSS_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ChromeRamSnapshot:
    chrome_process_count: int
    chrome_rss_bytes: int
    available_ram_bytes: int
    total_ram_bytes: int


def _format_gib(value: int) -> str:
    return f"{value / _GIB:.2f} GiB"


def _is_ram_wait(reason: str | None) -> bool:
    normalized = (reason or "").lower()

    markers = (
        "available ram",
        "reserved ram",
        "ram reserve",
        "memory pressure",
        "memory remained",
    )

    return any(marker in normalized for marker in markers)


def _sample_chrome_ram() -> ChromeRamSnapshot:
    memory = psutil.virtual_memory()

    chrome_process_count = 0
    chrome_rss_bytes = 0

    for process in psutil.process_iter(("name", "memory_info")):
        try:
            name = str(process.info.get("name") or "").lower()

            if name not in {"chrome", "chrome.exe"}:
                continue

            chrome_process_count += 1

            memory_info = process.info.get("memory_info")

            if memory_info is not None:
                chrome_rss_bytes += int(memory_info.rss)

        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue

    return ChromeRamSnapshot(
        chrome_process_count=chrome_process_count,
        chrome_rss_bytes=chrome_rss_bytes,
        available_ram_bytes=int(memory.available),
        total_ram_bytes=int(memory.total),
    )


def _admission_probe(watcher: TournamentWatcher) -> Callable[[TournamentWatchSnapshot], object | None]:
    """Create a passive fresh-assessment callback for the current candidate."""
    def assess(snapshot: TournamentWatchSnapshot) -> object | None:
        campaign = snapshot.campaign
        candidate_id = None if campaign is None else campaign.current_candidate_id
        if campaign is None or not candidate_id:
            return None
        try:
            manifest = watcher.store.tournaments.load_manifest(campaign.tournament_id)
            candidate = manifest.candidate(candidate_id)
            return watcher.resource_manager.assess(survival_resource_request(candidate))
        except Exception:
            return None

    return assess


_TERMINAL_CAMPAIGN_STATES = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
}

_TERMINAL_CANDIDATE_OUTCOMES = {
    "PASSED",
    "ELIMINATED",
    "BLOCKED",
    "CANCELLED",
}


def _value(value: object | None) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _line(symbol: str, title: str, detail: str | None = None) -> list[str]:
    lines = [f"[{symbol}] {title}"]
    if detail:
        for item in str(detail).splitlines():
            lines.append(f"    {item}")
    return lines


def _event_names(lines: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []

    for line in lines:
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue

        event = payload.get("event")
        if isinstance(event, str):
            values.append(event)

    return tuple(values)


def _candidate_symbol(
    outcome: str,
    *,
    current: bool,
    campaign_state: str,
) -> str:
    if outcome == "PASSED":
        return "\u2713"

    if outcome in {"ELIMINATED", "BLOCKED", "CANCELLED"}:
        return "!"

    if current and campaign_state == "WAITING_FOR_RESOURCES":
        return "~"

    if outcome == "RUNNING" or current:
        return ">"

    return " "


def _loop_warning(snapshot: TournamentWatchSnapshot, current: object | None) -> str | None:
    if current is None or snapshot.campaign is None:
        return None

    campaign_state = _value(snapshot.campaign.state)

    if campaign_state in _TERMINAL_CAMPAIGN_STATES:
        return None

    attempt_count = int(getattr(current, "attempt_count", 0) or 0)

    if attempt_count < 3:
        return None

    names = _event_names(snapshot.events)

    starts = sum(name == "survival started" for name in names)
    waits = sum(name == "waiting for resources" for name in names)

    if starts >= 2 and waits >= 2:
        return (
            f"Possible retry loop: candidate has reached attempt {attempt_count} "
            f"and recent history repeatedly alternates execution and resource waiting."
        )

    return None


def _next_action(
    snapshot: TournamentWatchSnapshot,
    current: object | None,
    *,
    worker_pid: int | None,
    owned_pid: int | None,
    loop_warning: str | None,
) -> str:
    campaign = snapshot.campaign

    if campaign is None:
        return "No tournament is available."

    state = _value(campaign.state)

    if state == "COMPLETED":
        return "Tournament is complete."

    if state == "CANCELLED":
        return "Tournament is cancelled. No further execution is expected."

    if state == "FAILED":
        return "Tournament has failed. Inspect the recorded reason."

    if loop_warning:
        return "Investigate the repeated step before allowing more execution attempts."

    if state == "WAITING_FOR_RESOURCES":
        retry = getattr(campaign, "next_retry_at", None)

        if retry is not None:
            return f"Wait for resources; next retry is scheduled for {retry.isoformat()}."

        return "Wait for sufficient resources before the next execution attempt."

    if worker_pid:
        return "Complete the active survival probe."

    if owned_pid:
        return "Start the benchmark worker against the owned Ollama runtime."

    if current is not None:
        provisioning = _value(getattr(current, "provisioning_state", None))

        if provisioning != "VERIFIED":
            return "Complete model provisioning and metadata verification."

        return "Perform resource admission and begin the survival attempt."

    return "Select the next unfinished candidate."


class TournamentChecklistRenderer:
    """Render one passive tournament snapshot as an operational checklist."""

    def __init__(
        self,
        *,
        ram_sampler: Callable[[], ChromeRamSnapshot] = _sample_chrome_ram,
        admission_assessor: Callable[[TournamentWatchSnapshot], object | None] | None = None,
    ) -> None:
        self._ram_sampler = ram_sampler
        self._admission_assessor = admission_assessor or (lambda snapshot: None)
        self._previous_ram_sample: ChromeRamSnapshot | None = None
        self._previous_ram_context: tuple[str, str] | None = None

    def _render_ram_wait_diagnostic(self, snapshot: TournamentWatchSnapshot) -> list[str]:
        sample = self._ram_sampler()
        assessment = self._admission_assessor(snapshot)
        decision = getattr(assessment, "decision", None)
        decision_text = _value(decision) or "UNAVAILABLE"
        headroom = getattr(assessment, "ram_headroom_bytes", None)
        shortfall = max(0, -headroom) if isinstance(headroom, int) else None
        conservative_reclaim = int(sample.chrome_rss_bytes * _CHROME_CONSERVATIVE_RECLAIM_RATIO)
        lines = [
            f"Chrome processes:          {sample.chrome_process_count}",
            f"Chrome RSS:                {_format_gib(sample.chrome_rss_bytes)}",
            f"Current available RAM:     {_format_gib(sample.available_ram_bytes)}",
            f"Total system RAM:          {_format_gib(sample.total_ram_bytes)}",
            f"Fresh resource admission:  {decision_text}",
        ]
        if assessment is not None:
            lines.extend((
                f"Configured RAM reserve:    {_format_gib(assessment.ram_reserved_bytes)}",
                f"Candidate RAM request:     {_format_gib(assessment.ram_required_bytes)}",
                ("Current RAM shortfall:      none" if shortfall == 0
                 else f"Current RAM shortfall:     {_format_gib(shortfall)}"),
            ))
        else:
            lines.append("Authoritative RAM shortfall is unavailable; projections are not classified.")

        chrome_negligible = sample.chrome_rss_bytes <= _CHROME_CLOSED_RSS_THRESHOLD_BYTES
        if chrome_negligible:
            lines.extend(("", "[-] Chrome is not currently consuming meaningful RAM."))
        elif shortfall is not None:
            lines.extend((
                f"Conservative reclaim estimate (60% RSS): {_format_gib(conservative_reclaim)}",
                f"Full RSS upper bound:                    {_format_gib(sample.chrome_rss_bytes)}",
                "    These are predictions, not guarantees of immediately reclaimable Windows RAM.",
                "",
            ))
            if decision is ResourceDecision.ALLOW or shortfall == 0:
                lines.extend((
                    "[~] Chrome is using RAM, but fresh authoritative admission no longer has a RAM shortfall.",
                    "    The scheduler may retry when its timer permits.",
                ))
            elif conservative_reclaim >= shortfall:
                lines.extend((
                    "[!] Chrome is likely a major contributor to this RAM wait.",
                    "[\u2713] Closing Chrome should likely resolve the current admission shortage.",
                    "    A fresh post-close assessment remains authoritative.",
                ))
            elif sample.chrome_rss_bytes >= shortfall:
                lines.extend((
                    "[~] Closing Chrome may resolve the RAM wait, but full RSS may not become immediately available.",
                    "    A fresh post-close assessment remains authoritative.",
                ))
            else:
                remaining = shortfall - sample.chrome_rss_bytes
                lines.extend((
                    "[!] Closing Chrome alone is unlikely to resolve the RAM wait.",
                    f"    Additional RAM still required after the full-RSS upper bound: about {_format_gib(remaining)}",
                ))

        previous = self._previous_ram_sample
        if (
            previous is not None
            and previous.chrome_rss_bytes >= _CHROME_SIGNIFICANT_RSS_BYTES
            and sample.chrome_rss_bytes <= _CHROME_CLOSED_RSS_THRESHOLD_BYTES
        ):
            gain = sample.available_ram_bytes - previous.available_ram_bytes
            sign = "+" if gain >= 0 else "-"
            change = "increased" if gain >= 0 else "decreased"
            lines.extend(("", "CHROME CHANGE DETECTED"))
            if sample.chrome_process_count == 0:
                lines.append("[\u2713] Chrome was closed.")
            else:
                lines.append("[\u2713] Chrome memory dropped to negligible background usage.")
            lines.append(f"    Available RAM {change} by {sign}{_format_gib(abs(gain))}.")
            if decision is ResourceDecision.ALLOW:
                lines.extend((
                    "[\u2713] The current resource-admission condition is resolved.",
                    "    T/TEA can retry when the scheduler/retry timer permits.",
                ))
            elif decision is ResourceDecision.DELAY:
                if gain > 0:
                    lines.append("[!] Closing Chrome improved available resources, but the resource condition is still not satisfied.")
                else:
                    lines.append("[!] Chrome memory dropped, but available RAM did not improve and the resource condition is still not satisfied.")
            elif decision is ResourceDecision.DENY:
                lines.append("[!] Fresh authoritative admission is DENY; inspect the recorded static constraint.")
            else:
                lines.append("[~] Post-close admission could not be assessed; the observed RAM increase is informational only.")
        self._previous_ram_sample = sample
        return lines

    def render(self, snapshot: TournamentWatchSnapshot) -> str:
        campaign = snapshot.campaign

        if campaign is None:
            return (
                "T/TEA TOURNAMENT CHECKLIST\n\n"
                "No survival tournament is available.\n\n"
                "Ctrl+C exits checklist only.\n"
            )

        completed, total = campaign.progress

        current_id = campaign.current_candidate_id or ""
        current = campaign.records.get(current_id)

        state = _value(campaign.state)
        outcome = _value(getattr(current, "outcome", None))
        provisioning = _value(getattr(current, "provisioning_state", None))

        benchmark = snapshot.benchmark_state or {}

        # Persisted benchmark state may retain the last worker/server PIDs
        # after cleanup. They are evidence, not proof that a process is
        # currently live. Never present them as active once either the
        # campaign or candidate is terminal, or when the supervisor itself
        # is no longer alive.
        runtime_live = (
            snapshot.supervisor_alive
            and state not in _TERMINAL_CAMPAIGN_STATES
            and outcome not in _TERMINAL_CANDIDATE_OUTCOMES
        )

        worker_pid = None
        owned_pid = None

        if current is not None and runtime_live:
            worker_pid = (
                benchmark.get("worker_pid")
                or getattr(current, "worker_pid", None)
            )
            owned_pid = (
                benchmark.get("owned_ollama_pid")
                or benchmark.get("owned_server_pid")
                or getattr(current, "owned_ollama_pid", None)
            )

        attempt_count = (
            int(getattr(current, "attempt_count", 0) or 0)
            if current is not None
            else 0
        )

        reason = (
            getattr(current, "reason", None)
            if current is not None
            else None
        ) or campaign.last_reason

        cleanup = (
            getattr(current, "cleanup_verified", None)
            if current is not None
            else None
        )

        recovery = (
            getattr(current, "resource_recovery_verified", None)
            if current is not None
            else None
        )

        isolation = (
            getattr(current, "default_server_identity_preserved", None)
            if current is not None
            else None
        )

        loop_warning = _loop_warning(snapshot, current)

        age_seconds = max(
            0.0,
            (snapshot.captured_at - campaign.updated_at).total_seconds(),
        )

        lines = [
            "T/TEA TOURNAMENT CHECKLIST",
            "",
            f"Tournament: {campaign.tournament_id}",
            f"State:      {state}",
            f"Round:      {campaign.current_round_id or '-'}",
            f"Candidate:  {current_id or '-'}",
            f"Progress:   {completed}/{total}",
            f"Supervisor: {'RUNNING' if snapshot.supervisor_alive else 'NOT RUNNING'}",
            "",
            "CURRENT CANDIDATE PIPELINE",
            "",
        ]

        if current is not None:
            lines += _line("\u2713", "Candidate selected", current_id)
        else:
            lines += _line(" ", "Candidate selected", "No active candidate.")

        if current is None:
            lines += _line(" ", "Model metadata / provisioning")
        elif provisioning == "VERIFIED":
            lines += _line(
                "\u2713",
                "Model metadata verified",
                f"{current.model} / provisioning state VERIFIED",
            )
        elif provisioning in {"BLOCKED", "FAILED"}:
            lines += _line(
                "!",
                "Model metadata / provisioning",
                f"{provisioning}: {reason or 'no reason recorded'}",
            )
        elif state in {"RUNNING", "WAITING_FOR_RESOURCES"}:
            lines += _line(
                ">",
                "Model metadata / provisioning",
                provisioning or "in progress",
            )
        else:
            lines += _line(
                " ",
                "Model metadata / provisioning",
                provisioning or "not reached",
            )

        if current is None:
            lines += _line(" ", "Resource admission")
        elif attempt_count > 0:
            lines += _line(
                "\u2713",
                "Resource admission",
                f"At least one real execution attempt was admitted. Attempt count: {attempt_count}",
            )
        elif state == "WAITING_FOR_RESOURCES":
            lines += _line(
                "~",
                "Resource admission",
                reason or "Waiting for sufficient resources.",
            )
        elif provisioning == "VERIFIED" and state == "RUNNING":
            lines += _line(
                ">",
                "Resource admission",
                "Checking fresh RAM / GPU / thermal conditions.",
            )
        else:
            lines += _line(" ", "Resource admission")

        if owned_pid:
            lines += _line(
                ">",
                "Dedicated Ollama runtime",
                f"Running with PID {owned_pid}",
            )
        elif attempt_count > 0 and state == "WAITING_FOR_RESOURCES":
            lines += _line(
                "\u2713",
                "Dedicated Ollama runtime",
                "Started during the previous attempt and is currently cleaned up.",
            )
        elif attempt_count > 0 and outcome in _TERMINAL_CANDIDATE_OUTCOMES:
            lines += _line(
                "\u2713",
                "Dedicated Ollama runtime",
                "No active owned runtime expected after terminal cleanup.",
            )
        elif provisioning == "VERIFIED" and state == "RUNNING":
            lines += _line(
                ">",
                "Dedicated Ollama runtime",
                "Starting or waiting for owned runtime startup.",
            )
        else:
            lines += _line(" ", "Dedicated Ollama runtime")

        if worker_pid:
            lines += _line(
                ">",
                "Benchmark worker",
                f"Running with PID {worker_pid}",
            )
        elif attempt_count > 0 and state == "WAITING_FOR_RESOURCES":
            lines += _line(
                "\u2713",
                "Benchmark worker",
                "Previous worker stopped after the last attempt.",
            )
        elif attempt_count > 0 and outcome in _TERMINAL_CANDIDATE_OUTCOMES:
            lines += _line(
                "\u2713",
                "Benchmark worker",
                "No active worker expected after terminal cleanup.",
            )
        elif owned_pid:
            lines += _line(
                ">",
                "Benchmark worker",
                "Waiting for worker startup.",
            )
        else:
            lines += _line(" ", "Benchmark worker")

        if current is None:
            lines += _line(" ", "Runtime resource safety")
        elif state == "WAITING_FOR_RESOURCES" and attempt_count > 0:
            lines += _line(
                "~",
                "Runtime resource safety",
                reason or "Execution paused because a runtime reserve could not be maintained.",
            )
        elif outcome == "RUNNING":
            lines += _line(
                ">",
                "Runtime resource safety",
                "RAM / VRAM / thermal reserves are being monitored.",
            )
        elif outcome == "PASSED":
            lines += _line("\u2713", "Runtime resource safety", "Survival probe completed safely.")
        elif outcome in {"ELIMINATED", "BLOCKED"}:
            lines += _line(
                "!",
                "Runtime resource safety / execution",
                reason or outcome,
            )
        elif outcome == "CANCELLED" or state == "CANCELLED":
            lines += _line(
                "-",
                "Runtime resource safety",
                "No active safety check; execution was cancelled.",
            )
        else:
            lines += _line(" ", "Runtime resource safety")

        if outcome == "PASSED":
            lines += _line("\u2713", "Survival response validated", "Candidate passed Round 1.")
        elif outcome in {"ELIMINATED", "BLOCKED"}:
            lines += _line(
                "!",
                "Survival response validated",
                reason or outcome,
            )
        elif outcome == "CANCELLED" or state == "CANCELLED":
            lines += _line(
                "-",
                "Survival response validation",
                "Not evaluated because execution was cancelled.",
            )
        elif worker_pid:
            lines += _line(
                ">",
                "Survival response validation",
                "Waiting for probe completion and validation.",
            )
        else:
            lines += _line(" ", "Survival response validation")

        if cleanup is True:
            lines += _line("\u2713", "Cleanup verified")
        elif cleanup is False:
            lines += _line("!", "Cleanup verified", "Cleanup verification failed.")
        elif attempt_count > 0 and state == "WAITING_FOR_RESOURCES":
            lines += _line(
                ">",
                "Cleanup verification",
                "Previous attempt ended; waiting for persisted cleanup evidence.",
            )
        else:
            lines += _line(" ", "Cleanup verification")

        if recovery is True:
            lines += _line("\u2713", "Resource recovery verified")
        elif recovery is False:
            lines += _line("!", "Resource recovery verified", "Recovery verification failed.")
        elif attempt_count > 0 and state == "WAITING_FOR_RESOURCES":
            lines += _line(
                ">",
                "Resource recovery verification",
                "Waiting for persisted recovery evidence.",
            )
        else:
            lines += _line(" ", "Resource recovery verification")

        if isolation is True:
            lines += _line("\u2713", "Default Ollama isolation preserved")
        elif isolation is False:
            lines += _line("!", "Default Ollama isolation preserved")
        else:
            lines += _line(" ", "Default Ollama isolation preserved")

        if outcome == "PASSED":
            lines += _line("\u2713", "Candidate decision", "PASS")
        elif outcome in {"ELIMINATED", "BLOCKED", "CANCELLED"}:
            lines += _line(
                "!",
                "Candidate decision",
                f"{outcome}: {reason or 'no reason recorded'}",
            )
        elif state == "WAITING_FOR_RESOURCES":
            lines += _line(
                "~",
                "Candidate decision",
                "Waiting for the resource condition before continuing.",
            )
        elif outcome == "RUNNING":
            lines += _line(">", "Candidate decision", "Survival execution is still active.")
        else:
            lines += _line(" ", "Candidate decision")

        if state == "COMPLETED":
            lines += _line("\u2713", "Advance / finish tournament", "Tournament completed.")
        elif state == "CANCELLED":
            lines += _line("-", "Advance to next candidate", "Tournament was cancelled.")
        elif outcome in _TERMINAL_CANDIDATE_OUTCOMES:
            lines += _line(
                ">",
                "Advance to next candidate",
                "Current candidate is terminal; scheduler should move forward.",
            )
        else:
            lines += _line(
                " ",
                "Advance to next candidate",
                "Waiting for the current candidate decision.",
            )

        lines.extend(("", "SURVIVAL CANDIDATES", ""))

        for candidate_id, record in campaign.records.items():
            record_outcome = _value(record.outcome)

            if (
                candidate_id == current_id
                and state == "WAITING_FOR_RESOURCES"
                and record_outcome in {"PENDING", "WAITING_FOR_RESOURCES"}
            ):
                label = "WAITING"
            else:
                label = record_outcome or "PENDING"

            symbol = _candidate_symbol(
                record_outcome,
                current=candidate_id == current_id,
                campaign_state=state,
            )

            suffix = ""

            if candidate_id == current_id:
                suffix = f"  <- current / attempt {record.attempt_count}"

            lines.append(
                f"[{symbol}] {candidate_id:<25} {label}{suffix}"
            )

        if (
            state == "WAITING_FOR_RESOURCES"
            and _is_ram_wait(reason)
        ):
            ram_context = (campaign.tournament_id, current_id)
            if self._previous_ram_context != ram_context:
                self._previous_ram_sample = None
            self._previous_ram_context = ram_context
            lines.extend(("", "RAM WAIT DIAGNOSTIC", ""))
            lines.extend(self._render_ram_wait_diagnostic(snapshot))

        if state != "WAITING_FOR_RESOURCES" or not _is_ram_wait(reason):
            self._previous_ram_sample = None
            self._previous_ram_context = None

        lines.extend(("", "STATUS", ""))

        if loop_warning:
            lines += _line("!", "POSSIBLE RETRY LOOP", loop_warning)
        elif state == "WAITING_FOR_RESOURCES":
            lines += _line(
                "~",
                "WAITING FOR RESOURCES",
                reason or "No detailed resource reason was recorded.",
            )
        elif state == "RUNNING":
            lines += _line(">", "RUNNING")
        elif state == "COMPLETED":
            lines += _line("\u2713", "COMPLETED")
        elif state in {"FAILED", "CANCELLED"}:
            lines += _line("!", state, campaign.last_reason)
        else:
            lines += _line(" ", state)

        lines.extend(
            (
                "",
                f"Next expected action: {_next_action(snapshot, current, worker_pid=worker_pid, owned_pid=owned_pid, loop_warning=loop_warning)}",
                f"Last persisted state change: {age_seconds:.1f}s ago",
                "",
                "Legend: [\u2713] done  [>] active  [~] waiting  [!] blocked/problem  [ ] not reached  [-] not applicable",
                "Ctrl+C exits checklist only.",
            )
        )

        return "\n".join(lines) + "\n"


def checklist(
    root: Path,
    *,
    refresh_seconds: float = 2.0,
    once: bool = False,
    output: TextIO = sys.stdout,
    sleep: Callable[[float], None] = time.sleep,
    watcher: TournamentWatcher | None = None,
) -> int:
    if refresh_seconds < 0.5:
        raise ValueError("refresh must be at least 0.5 seconds")

    watcher = watcher or TournamentWatcher(root)
    admission_assessor = _admission_probe(watcher) if isinstance(watcher, TournamentWatcher) else None
    renderer = TournamentChecklistRenderer(admission_assessor=admission_assessor)

    try:
        while True:
            snapshot = watcher.snapshot()

            if not once and getattr(output, "isatty", lambda: False)():
                output.write("\x1b[2J\x1b[H")

            output.write(renderer.render(snapshot))
            output.flush()

            if once:
                return 0

            sleep(refresh_seconds)

    except KeyboardInterrupt:
        return 0
