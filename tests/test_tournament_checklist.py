from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil

from ttea.resources import ResourceDecision
from ttea.tournament import CandidateVariant, TournamentCandidate
from ttea.tournament.checklist_watch import (
    ChromeRamSnapshot,
    TournamentChecklistRenderer,
    _admission_probe,
    _sample_chrome_ram,
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
    candidate_id: str = "qwen35-08b",
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
        candidate_id,
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
        current_candidate_id=candidate_id,
        current_round_id="survival_4k",
        records={
            candidate_id: current,
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


def admission(decision: ResourceDecision, shortfall_bytes: int = 0):
    return SimpleNamespace(
        decision=decision,
        ram_headroom_bytes=-shortfall_bytes,
        ram_reserved_bytes=4 * 1024**3,
        ram_required_bytes=512 * 1024**2,
        reasons=("current available RAM cannot satisfy requirement plus reserve",),
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

    def test_large_chrome_footprint_likely_resolves_ram_wait(self):
        gib = 1024**3
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: ChromeRamSnapshot(15, 2 * gib, 2 * gib, 16 * gib),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, gib),
        )
        text = renderer.render(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="current available RAM cannot satisfy requirement plus reserve",
        ))
        self.assertIn("Chrome processes:          15", text)
        self.assertIn("likely a major contributor", text)
        self.assertIn("should likely resolve", text)
        self.assertIn("predictions, not guarantees", text)

    def test_chrome_full_rss_only_crossing_shortfall_may_resolve(self):
        gib = 1024**3
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: ChromeRamSnapshot(8, int(1.5 * gib), 2 * gib, 16 * gib),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, gib),
        )
        text = renderer.render(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM is below reserve",
        ))
        self.assertIn("may resolve the RAM wait", text)
        self.assertIn("full RSS may not become immediately available", text)

    def test_chrome_insufficient_reports_remaining_shortfall(self):
        gib = 1024**3
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: ChromeRamSnapshot(5, gib, gib, 16 * gib),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, 2 * gib),
        )
        text = renderer.render(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="memory pressure below RAM reserve",
        ))
        self.assertIn("Closing Chrome alone is unlikely", text)
        self.assertIn("Additional RAM still required", text)
        self.assertIn("1.00 GiB", text)

    def test_absent_chrome_is_not_significant(self):
        gib = 1024**3
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: ChromeRamSnapshot(0, 0, 2 * gib, 16 * gib),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, gib),
        )
        text = renderer.render(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        ))
        self.assertIn("Chrome is not currently consuming meaningful RAM", text)
        self.assertNotIn("should likely resolve", text)

    def test_consecutive_samples_detect_chrome_close_and_actual_ram_gain(self):
        gib = 1024**3
        samples = iter((
            ChromeRamSnapshot(12, 2 * gib, 2 * gib, 16 * gib),
            ChromeRamSnapshot(0, 0, 3 * gib, 16 * gib),
        ))
        assessments = iter((
            admission(ResourceDecision.DELAY, gib),
            admission(ResourceDecision.ALLOW),
        ))
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: next(samples),
            admission_assessor=lambda _: next(assessments),
        )
        waiting = snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        )
        renderer.render(waiting)
        text = renderer.render(waiting)
        self.assertIn("CHROME CHANGE DETECTED", text)
        self.assertIn("Chrome was closed", text)
        self.assertIn("Available RAM increased by +1.00 GiB", text)
        self.assertIn("resource-admission condition is resolved", text)
        self.assertIn("scheduler/retry timer permits", text)

    def test_post_close_fresh_delay_is_explicitly_unresolved(self):
        gib = 1024**3
        samples = iter((
            ChromeRamSnapshot(10, 2 * gib, gib, 16 * gib),
            ChromeRamSnapshot(1, 32 * 1024**2, 2 * gib, 16 * gib),
        ))
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: next(samples),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, gib),
        )
        waiting = snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        )
        renderer.render(waiting)
        text = renderer.render(waiting)
        self.assertIn("CHROME CHANGE DETECTED", text)
        self.assertIn("resource condition is still not satisfied", text)

    def test_chrome_close_with_lower_available_ram_does_not_claim_improvement(self):
        gib = 1024**3
        samples = iter((
            ChromeRamSnapshot(10, 2 * gib, 3 * gib, 16 * gib),
            ChromeRamSnapshot(0, 0, 2 * gib, 16 * gib),
        ))
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: next(samples),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, gib),
        )
        waiting = snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        )
        renderer.render(waiting)
        text = renderer.render(waiting)
        self.assertIn("Available RAM decreased by -1.00 GiB", text)
        self.assertIn("available RAM did not improve", text)
        self.assertNotIn("improved available resources", text)

    def test_chrome_sample_does_not_carry_between_candidates(self):
        gib = 1024**3
        samples = iter((
            ChromeRamSnapshot(10, 2 * gib, gib, 16 * gib),
            ChromeRamSnapshot(0, 0, 2 * gib, 16 * gib),
        ))
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: next(samples),
            admission_assessor=lambda _: admission(ResourceDecision.DELAY, gib),
        )
        renderer.render(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        ))
        text = renderer.render(snapshot(
            candidate_id="qwen35-4b",
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        ))
        self.assertNotIn("CHROME CHANGE DETECTED", text)

    def test_post_close_fresh_deny_reports_static_constraint(self):
        gib = 1024**3
        samples = iter((
            ChromeRamSnapshot(10, 2 * gib, gib, 16 * gib),
            ChromeRamSnapshot(0, 0, 2 * gib, 16 * gib),
        ))
        renderer = TournamentChecklistRenderer(
            ram_sampler=lambda: next(samples),
            admission_assessor=lambda _: admission(ResourceDecision.DENY, gib),
        )
        waiting = snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        )
        renderer.render(waiting)
        text = renderer.render(waiting)
        self.assertIn("Fresh authoritative admission is DENY", text)
        self.assertIn("inspect the recorded static constraint", text)

    def test_chrome_sampling_aggregates_processes_and_ignores_access_errors(self):
        class InaccessibleProcess:
            @property
            def info(self):
                raise psutil.AccessDenied(pid=30)

        processes = (
            SimpleNamespace(info={
                "name": "chrome.exe",
                "memory_info": SimpleNamespace(rss=100),
            }),
            InaccessibleProcess(),
            SimpleNamespace(info={
                "name": "Chrome",
                "memory_info": SimpleNamespace(rss=250),
            }),
            SimpleNamespace(info={
                "name": "not-chrome.exe",
                "memory_info": SimpleNamespace(rss=999),
            }),
        )
        memory = SimpleNamespace(available=1_000, total=2_000)
        with (
            patch("ttea.tournament.checklist_watch.psutil.virtual_memory", return_value=memory),
            patch("ttea.tournament.checklist_watch.psutil.process_iter", return_value=processes),
        ):
            sample = _sample_chrome_ram()
        self.assertEqual(sample.chrome_process_count, 2)
        self.assertEqual(sample.chrome_rss_bytes, 350)
        self.assertEqual(sample.available_ram_bytes, 1_000)

    def test_non_ram_wait_skips_chrome_diagnosis(self):
        sampler = Mock(side_effect=AssertionError("Chrome sampling should not run"))
        renderer = TournamentChecklistRenderer(ram_sampler=sampler)
        text = renderer.render(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="GPU temperature is required but unavailable",
        ))
        self.assertNotIn("RAM WAIT DIAGNOSTIC", text)
        sampler.assert_not_called()

    def test_default_probe_uses_manifest_candidate_and_resource_manager(self):
        manifest = Mock()
        manager = Mock()
        expected = admission(ResourceDecision.DELAY, 1024)
        manager.assess.return_value = expected
        watcher = SimpleNamespace(
            store=SimpleNamespace(tournaments=SimpleNamespace(load_manifest=Mock(return_value=manifest))),
            resource_manager=manager,
        )
        real_candidate = TournamentCandidate(
            "qwen35-08b", "qwen", CandidateVariant("ollama", "qwen:tag"),
            1, 1, 1, 4096, "test",
        )
        manifest.candidate.return_value = real_candidate
        result = _admission_probe(watcher)(snapshot(
            state="WAITING_FOR_RESOURCES", outcome="PENDING",
            reason="available RAM cannot satisfy reserve",
        ))
        self.assertIs(result, expected)
        manifest.candidate.assert_called_once_with("qwen35-08b")
        request = manager.assess.call_args.args[0]
        self.assertEqual(request.workload_type, "tournament-survival:qwen35-08b")
        self.assertEqual(request.estimated_ram_bytes, 512 * 1024**2)

    def test_checklist_source_has_no_process_termination_calls(self):
        source = (Path(__file__).parents[1] / "src" / "ttea" /
                  "tournament" / "checklist_watch.py").read_text(encoding="utf-8").lower()
        for forbidden in (".terminate(", ".kill(", "taskkill", "killall"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
