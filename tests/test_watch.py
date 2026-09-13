from datetime import datetime, timedelta, timezone
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ttea.validation.campaign import CampaignStore
from ttea.validation.campaign_models import CampaignRunRecord, CampaignState, ValidationCampaign
from ttea.validation.models import ValidationStatus
from ttea.validation.watch import (CampaignWatchReader, CampaignWatchRenderer, CampaignWatchSnapshot,
                                   CampaignWatcher, bounded_tail, human_bytes, human_duration)


NOW = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)


def campaign(state=CampaignState.COMPLETED, history=()):
    return ValidationCampaign("campaign-watch-test", ("pre_model", "tiny_model"), "tiny_model", "tiny_model",
                              state, NOW - timedelta(minutes=7), NOW, NOW + timedelta(hours=1),
                              99, 1.0, NOW, "run-tiny", ValidationStatus.PASS, "completed", 2,
                              None, 2.0, False, True, None, tuple(history), ())


def ollama():
    return {"executable": "C:/Ollama/ollama.exe", "version": "ollama version 1", "default_running": True,
            "default_pid": 10, "dedicated_running": False, "owned_pid": None,
            "model": "smollm2:135m-instruct-q4_0", "installed": {"digest": "sha256:test", "size": 1},
            "model_running": False, "api_error": None}


def telemetry():
    return {"cpu_percent": 8.4, "ram_available_bytes": 3 * 1024**3, "ram_reserved_bytes": 1 * 1024**3,
            "ram_allocatable_bytes": 2 * 1024**3, "disk_free_bytes": 100 * 1024**3,
            "gpus": [{"index": 0, "name": "NVIDIA", "utilization_percent": 4,
                      "free_vram_bytes": 2 * 1024**3, "reserved_vram_bytes": 256 * 1024**2,
                      "allocatable_vram_bytes": 1792 * 1024**2, "temperature_c": 60,
                      "max_temperature_c": 80}]}


class WatchTests(unittest.TestCase):
    def snapshot(self, state=CampaignState.COMPLETED, *, report=True, results=()):
        payload = {"safety_summary": {"model": "smollm2:135m-instruct-q4_0", "parameter_size": "134.52M",
                   "quantization": "Q4_0", "tokens_per_second": 110.717, "eval_count": 16,
                   "gpu_max_temperature_c": 64, "server_cleanup_confirmed": True,
                   "resources_recovered": True, "forced_timeout_outcome": "TIMEOUT"}} if report else None
        return CampaignWatchSnapshot(campaign(state), False, len(results), 19, tuple(results), payload,
                                     ("2026-09-13T09:00:00+00:00 campaign completed",), None,
                                     telemetry(), None, ollama(), NOW)

    def test_humanization_and_bounded_tail(self):
        self.assertEqual(human_bytes(1536 * 1024**2), "1.50 GiB")
        self.assertEqual(human_duration(65), "00:01:05")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.log"
            path.write_text("\n".join(f"event {index}" for index in range(100)), encoding="utf-8")
            self.assertEqual(bounded_tail(path, 3, 32), ("event 97", "event 98", "event 99"))

    def test_completed_waiting_running_failed_and_interrupted_rendering(self):
        renderer = CampaignWatchRenderer(color=False, width=100)
        completed = renderer.render(self.snapshot(results=({"status": "PASS", "scenario_id": "tiny_model_inference"},)))
        self.assertIn("COMPLETED", completed)
        self.assertIn("134.52M", completed)
        self.assertIn("127.0.0.1:11434", completed)
        waiting = self.snapshot(CampaignState.WAITING_FOR_RESOURCES)
        waiting = replace(waiting, campaign=replace(waiting.campaign, next_retry_at=NOW + timedelta(seconds=30),
                                                     last_reason="low RAM"))
        self.assertIn("WAITING FOR RESOURCES", renderer.render(waiting))
        for state in (CampaignState.RUNNING, CampaignState.FAILED, CampaignState.INTERRUPTED):
            self.assertIn(state.value, renderer.render(self.snapshot(state)))

    def test_narrow_unavailable_and_no_campaign_rendering(self):
        renderer = CampaignWatchRenderer(color=False, width=28)
        value = self.snapshot()
        unavailable = replace(value, telemetry=None, telemetry_error="sensor failure")
        text = renderer.render(unavailable)
        self.assertIn("Telemetry unavailable", text)
        self.assertTrue(all(len(line) <= 28 for line in text.splitlines()))
        none = CampaignWatchSnapshot(None, False, 0, 0, (), None, (), None, None, None, ollama(), NOW)
        self.assertIn("No validation campaign", renderer.render(none))

    def test_reader_progress_ignores_temporary_blocked_and_keeps_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, reports = Path(directory) / "campaigns", Path(directory) / "reports"
            record = CampaignRunRecord("tiny_model", "run-tiny", ValidationStatus.BLOCKED, "low RAM", NOW, NOW, True)
            value = campaign(CampaignState.WAITING_FOR_RESOURCES, (record,))
            store = CampaignStore(root)
            store.create(value)
            report_dir = reports / "run-tiny"
            report_dir.mkdir(parents=True)
            report_dir.joinpath("report.json").write_text(json.dumps({"scenario_results": [
                {"scenario_id": "tiny_model_manifest", "status": "PASS"},
                {"scenario_id": "tiny_model_inference", "status": "BLOCKED"}]}), encoding="utf-8")
            before = store.state_path(value.campaign_id).read_bytes()
            reader = CampaignWatchReader(root, reports, Path(directory) / "state.json", telemetry_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
            with patch.object(reader, "_ollama", return_value=ollama()):
                snapshot = reader.read()
            self.assertEqual((snapshot.progress_completed, snapshot.progress_total), (1, 19))
            self.assertEqual(snapshot.telemetry_error, "offline")
            self.assertEqual(before, store.state_path(value.campaign_id).read_bytes())

    def test_watcher_once_json_and_ctrl_c_are_read_only(self):
        reader = Mock()
        reader.read.return_value = self.snapshot()
        output = StringIO()
        watcher = CampaignWatcher(reader, CampaignWatchRenderer(color=False), output=output)
        self.assertEqual(watcher.run(once=True, json_output=True), 0)
        self.assertIn('"progress"', output.getvalue())
        reader.read.side_effect = KeyboardInterrupt
        self.assertEqual(watcher.run(), 0)

    def test_no_color_tty_redraws_without_color_codes_and_terminal_elapsed_is_stable(self):
        snapshot = self.snapshot()
        later = replace(snapshot, captured_at=NOW + timedelta(days=1))
        renderer = CampaignWatchRenderer(color=False, width=80)
        self.assertIn("Elapsed        00:07:00", renderer.render(snapshot))
        self.assertIn("Elapsed        00:07:00", renderer.render(later))
        reader, output = Mock(), Mock()
        reader.read.side_effect = [snapshot, KeyboardInterrupt]
        output.isatty.return_value = True
        watcher = CampaignWatcher(reader, renderer, output=output, redraw_supported=True, sleep=lambda _: None)
        self.assertEqual(watcher.run(), 0)
        written = "".join(call.args[0] for call in output.write.call_args_list)
        self.assertIn("\x1b[2J\x1b[H", written)
        self.assertNotIn("\x1b[3", written)

    def test_latest_result_uses_history_not_current_suite(self):
        record = CampaignRunRecord("pre_model", "run-pre", ValidationStatus.PASS, "done", NOW, NOW, True)
        value = self.snapshot(CampaignState.RUNNING)
        value = replace(value, campaign=campaign(CampaignState.RUNNING, (record,)))
        text = CampaignWatchRenderer(color=False).render(value)
        self.assertRegex(
            text,
            r"Latest result\s+PASS pre_model",
        )

    def test_ollama_metadata_is_cached_and_reader_never_calls_active_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaigns"
            store = CampaignStore(root)
            store.create(campaign())
            reader = CampaignWatchReader(root, Path(directory) / "reports", Path(directory) / "state.json",
                                         telemetry_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
            with patch("ttea.validation.watch.resolve_ollama_executable", return_value="ollama.exe"), \
                 patch("ttea.validation.watch.subprocess.run", return_value=Mock(returncode=0, stdout="v", stderr="")) as version, \
                 patch("ttea.validation.watch.loopback_port_open", return_value=False), \
                 patch("ttea.validation.watch.OllamaClient.pull") as pull, \
                 patch("ttea.validation.watch.OllamaClient.generate") as generate:
                reader.read(); reader.read()
            version.assert_called_once()
            pull.assert_not_called(); generate.assert_not_called()

    def test_cli_watch_parsing_and_minimum_refresh(self):
        from ttea.validation import __main__
        with patch("sys.argv", ["ttea.validation", "campaign", "watch", "--once", "--no-color"]), \
             patch("ttea.validation.__main__.CampaignWatcher.run", return_value=0) as run:
            __main__.main()
        self.assertTrue(run.call_args.kwargs["once"])
        with patch("sys.argv", ["ttea.validation", "campaign", "watch", "--refresh", "0.1"]):
            with self.assertRaises(SystemExit):
                __main__.main()


if __name__ == "__main__":
    unittest.main()
