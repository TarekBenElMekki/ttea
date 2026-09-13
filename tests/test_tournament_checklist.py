from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

from ttea.tournament.checklist_watch import (
    TournamentChecklistRenderer,
    checklist,
)


NOW = datetime(2026, 9, 13, 16, 0, tzinfo=timezone.utc)


def enum(value: str):
    return SimpleNamespace(value=value)


def record(
    candidate_id: str = "qwen35-08b",
    *,
    outcome: str = "PENDING",
    attempt_count: int = 0,
    reason: str | None = None,
    worker_pid: int | None = None,
    owned_pid: int | None = None,
    cleanup=None,
    recovery=None,
):
    return SimpleNamespace(
        candidate_id=candidate_id,
        model="qwen3.5:0.8b",
        provisioning_state=enum("VERIFIED"),
        outcome=enum(outcome),
        attempt_count=attempt_count,
        reason=reason,
        worker_pid=worker_pid,
        owned_ollama_pid=owned_pid,
        cleanup_verified=cleanup,
        resource_recovery_verified=recovery,
        default_server_identity_preserved=True,
    )


def snapshot(
    *,
    state: str = "RUNNING",
    outcome: str = "RUNNING",
    attempt_count: int = 1,
    reason: str | None = None,
    worker_pid: int | None = None,
    owned_pid: int | None = None,
    cleanup=None,
    recovery=None,
    events=(),
):
    current = record(
        outcome=outcome,
        attempt_count=attempt_count,
        reason=reason,
        worker_pid=worker_pid,
        owned_pid=owned_pid,
        cleanup=cleanup,
        recovery=recovery,
    )

    next_record = record(
        "qwen35-2b",
        outcome="PENDING",
    )

    campaign = SimpleNamespace(
        tournament_id="tournament-test",
        state=enum(state),
        current_candidate_id="qwen35-08b",
        current_round_id="survival_4k",
        records={
            "qwen35-08b": current,
            "qwen35-2b": next_record,
        },
        progress=(0, 2),
        created_at=NOW,
        updated_at=NOW,
        next_retry_at=NOW + timedelta(seconds=30),
        last_event=None,
        last_reason=reason,
    )

    benchmark = {
        "worker_pid": worker_pid,
        "owned_ollama_pid": owned_pid,
    }

    return SimpleNamespace(
        campaign=campaign,
        supervisor_alive=state not in {"CANCELLED", "COMPLETED", "FAILED"},
        captured_at=NOW + timedelta(seconds=10),
        benchmark_state=benchmark,
        events=tuple(events),
    )


class TournamentChecklistTests(unittest.TestCase):
    def test_running_worker_is_shown_as_active(self):
        text = TournamentChecklistRenderer().render(
            snapshot(
                worker_pid=1234,
                owned_pid=5678,
            )
        )

        self.assertIn("[>] Dedicated Ollama runtime", text)
        self.assertIn("PID 5678", text)
        self.assertIn("[>] Benchmark worker", text)
        self.assertIn("PID 1234", text)
        self.assertIn("[>] Runtime resource safety", text)

    def test_waiting_for_resources_is_explicit(self):
        text = TournamentChecklistRenderer().render(
            snapshot(
                state="WAITING_FOR_RESOURCES",
                outcome="PENDING",
                attempt_count=1,
                reason="available RAM remained below the reserved RAM threshold",
            )
        )

        self.assertIn("[~] Runtime resource safety", text)
        self.assertIn("WAITING FOR RESOURCES", text)
        self.assertIn(
            "available RAM remained below the reserved RAM threshold",
            text,
        )
        self.assertIn("Next expected action:", text)

    def test_repeated_resource_attempts_show_possible_loop(self):
        event_lines = (
            '{"event":"survival started"}',
            '{"event":"waiting for resources"}',
            '{"event":"survival started"}',
            '{"event":"waiting for resources"}',
        )

        text = TournamentChecklistRenderer().render(
            snapshot(
                state="WAITING_FOR_RESOURCES",
                outcome="PENDING",
                attempt_count=8,
                reason="available RAM remained below the reserved RAM threshold",
                events=event_lines,
            )
        )

        self.assertIn("POSSIBLE RETRY LOOP", text)
        self.assertIn("attempt 8", text)
        self.assertIn(
            "Investigate the repeated step before allowing more execution attempts.",
            text,
        )

    def test_cancelled_campaign_is_terminal_and_does_not_claim_running(self):
        text = TournamentChecklistRenderer().render(
            snapshot(
                state="CANCELLED",
                outcome="CANCELLED",
                attempt_count=8,
                reason="tournament cancellation requested",
                worker_pid=13544,
                owned_pid=14704,
            )
        )

        self.assertIn("[!] CANCELLED", text)
        self.assertIn("Supervisor: NOT RUNNING", text)
        self.assertIn("Tournament is cancelled", text)

        # Persisted benchmark evidence may contain stale PIDs after cleanup.
        # Terminal campaigns must never present those PIDs as live processes.
        self.assertNotIn("Running with PID 13544", text)
        self.assertNotIn("Running with PID 14704", text)

        self.assertIn(
            "No active owned runtime expected after terminal cleanup.",
            text,
        )
        self.assertIn(
            "No active worker expected after terminal cleanup.",
            text,
        )
        self.assertIn(
            "No active safety check; execution was cancelled.",
            text,
        )
        self.assertIn(
            "Not evaluated because execution was cancelled.",
            text,
        )

    def test_checklist_once_is_read_only_and_uses_existing_watcher(self):
        watcher = Mock()
        watcher.snapshot.return_value = snapshot()

        output = StringIO()

        result = checklist(
            SimpleNamespace(),
            once=True,
            output=output,
            watcher=watcher,
        )

        self.assertEqual(result, 0)
        self.assertIn("T/TEA TOURNAMENT CHECKLIST", output.getvalue())
        self.assertIn("[\u2713] Candidate selected", output.getvalue())
        self.assertNotIn("\u00e2\u0153\u201c", output.getvalue())
        watcher.snapshot.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()