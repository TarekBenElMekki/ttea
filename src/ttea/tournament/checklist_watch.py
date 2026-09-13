"""Read-only checklist-style terminal monitor for survival tournaments."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

from .watch import TournamentWatchSnapshot, TournamentWatcher


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
    renderer = TournamentChecklistRenderer()

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