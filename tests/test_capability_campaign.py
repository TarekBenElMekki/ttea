from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ttea.tournament.capability import (
    ROUND2_CANDIDATES,
    ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST,
    ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID,
)
from ttea.tournament.capability_campaign import (
    CapabilityCampaign,
    CapabilityCampaignManager,
    CapabilityCampaignState,
    CapabilityCampaignStore,
    capability_report_payload,
    capability_report_text,
)
from ttea.tournament.catalog import build_survival_manifest
from ttea.tournament.execution import (
    ProvisioningState,
    SurvivalCampaignState,
    SurvivalOutcome,
)
from ttea.tournament.__main__ import main as tournament_main


NOW = datetime.now(timezone.utc)


class FakeProcess:
    def create_time(self):
        return 12.5


class CapabilityCampaignTests(unittest.TestCase):
    def campaign(self, tournament_id="cap-test", state=CapabilityCampaignState.CREATED):
        return CapabilityCampaign(
            tournament_id,
            ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID,
            ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST,
            "derived-digest",
            state,
            NOW,
            NOW,
            None,
            None,
            "test campaign",
        )

    def test_campaign_store_round_trip_progress_and_terminal_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CapabilityCampaignStore(Path(directory))
            campaign = self.campaign()
            store.create(campaign)
            loaded = store.load(campaign.tournament_id)
            self.assertEqual(loaded, campaign)
            self.assertEqual(store.progress(campaign.tournament_id)[:2], (0, 36))
            terminal = store.save(replace(
                campaign,
                state=CapabilityCampaignState.COMPLETED_INCOMPLETE,
                updated_at=NOW,
                last_reason="finished",
            ))
            with self.assertRaises(Exception):
                store.save(replace(terminal, last_reason="rewritten"))

    def test_report_hierarchy_uses_refs_not_copied_raw_response(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CapabilityCampaignStore(Path(directory))
            campaign = self.campaign(state=CapabilityCampaignState.INTERRUPTED)
            store.create(campaign)
            payload = capability_report_payload(campaign, store)
            text = capability_report_text(campaign, store)
        self.assertEqual(payload["progress"]["total"], 36)
        self.assertEqual(len(payload["work_evidence"]), 36)
        self.assertNotIn("response_utf8", json.dumps(payload))
        self.assertNotIn("parsed_response", json.dumps(payload))
        self.assertIn("Capability Round 2", text)

    def test_status_reconciles_stale_running_owner_to_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = CapabilityCampaignManager(
                Path(directory),
                process_api=lambda _pid: FakeProcess(),
            )
            running = replace(
                self.campaign(state=CapabilityCampaignState.RUNNING),
                owner_pid=999999,
                owner_started_epoch=1.0,
            )
            manager.store.create(running)
            with patch.object(manager.store, "owner_alive", return_value=False):
                campaign, alive = manager.status(running.tournament_id)
        self.assertFalse(alive)
        self.assertEqual(campaign.state, CapabilityCampaignState.INTERRUPTED)

    def test_cli_capability_start_rejects_download_and_detach_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with redirect_stdout(output):
                code_download = tournament_main([
                    "--root", str(root), "start", "--through", "capability",
                    "--allow-model-download",
                ])
            self.assertEqual(code_download, 2)
            self.assertIn("never authorizes model download", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                code_detach = tournament_main([
                    "--root", str(root), "start", "--through", "capability",
                    "--detach",
                ])
            self.assertEqual(code_detach, 2)
            self.assertIn("detached supervision is not enabled", output.getvalue())

    def test_cli_capability_status_and_report_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CapabilityCampaignStore(root)
            campaign = self.campaign(state=CapabilityCampaignState.INTERRUPTED)
            store.create(campaign)
            before = store.campaign_path(campaign.tournament_id).read_bytes()

            output = io.StringIO()
            with redirect_stdout(output):
                status_code = tournament_main([
                    "--root", str(root), "status", campaign.tournament_id,
                ])
            self.assertEqual(status_code, 0)
            self.assertIn("CAPABILITY_ROUND_2", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                report_code = tournament_main([
                    "--root", str(root), "report", campaign.tournament_id,
                ])
            self.assertEqual(report_code, 0)
            self.assertIn("Capability Round 2", output.getvalue())
            self.assertEqual(
                store.campaign_path(campaign.tournament_id).read_bytes(),
                before,
            )



    def test_source_campaign_requires_all_four_verified_survivors(self):
        manager = CapabilityCampaignManager(
            Path("."),
            process_api=lambda _pid: FakeProcess(),
        )

        records = {
            candidate_id: SimpleNamespace(
                outcome=SurvivalOutcome.PASSED,
                provisioning_state=ProvisioningState.VERIFIED,
                provisioning=SimpleNamespace(candidate_id=candidate_id),
            )
            for candidate_id in ROUND2_CANDIDATES
        }
        source = SimpleNamespace(
            tournament_id=ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID,
            manifest_digest=ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST,
            state=SurvivalCampaignState.COMPLETED,
            records=records,
        )
        manager.survival_store = SimpleNamespace(load=lambda _tournament_id: source)

        resolved = manager._source_campaign()
        self.assertIs(resolved, source)

        bad_records = dict(records)
        bad_records[ROUND2_CANDIDATES[0]] = SimpleNamespace(
            outcome=SurvivalOutcome.ELIMINATED,
            provisioning_state=ProvisioningState.VERIFIED,
            provisioning=SimpleNamespace(candidate_id=ROUND2_CANDIDATES[0]),
        )
        manager.survival_store = SimpleNamespace(
            load=lambda _tournament_id: SimpleNamespace(
                tournament_id=ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID,
                manifest_digest=ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST,
                state=SurvivalCampaignState.COMPLETED,
                records=bad_records,
            )
        )
        with self.assertRaisesRegex(Exception, "did not pass"):
            manager._source_campaign()

    def test_preflight_verifies_exact_manifest_candidates_with_persisted_survival_evidence(self):
        manager = CapabilityCampaignManager(
            Path("."),
            process_api=lambda _pid: FakeProcess(),
        )
        manifest = build_survival_manifest()
        survivors = tuple(
            manifest.candidate(candidate_id)
            for candidate_id in ROUND2_CANDIDATES
        )

        prior_by_id = {
            candidate_id: object()
            for candidate_id in ROUND2_CANDIDATES
        }
        source = SimpleNamespace(
            records={
                candidate_id: SimpleNamespace(provisioning=prior_by_id[candidate_id])
                for candidate_id in ROUND2_CANDIDATES
            }
        )

        calls = []

        class FakeProvider:
            def provision(self, candidate, prior=None):
                calls.append((candidate, prior))
                return prior

        manager._preflight_source_models(
            FakeProvider(),
            source,
            SimpleNamespace(
                candidate=lambda candidate_id: next(
                    candidate
                    for candidate in survivors
                    if candidate.candidate_id == candidate_id
                )
            ),
        )

        self.assertEqual(
            tuple(candidate.candidate_id for candidate, _prior in calls),
            ROUND2_CANDIDATES,
        )
        for (candidate, prior), expected_id in zip(calls, ROUND2_CANDIDATES):
            self.assertIs(
                candidate,
                next(item for item in survivors if item.candidate_id == expected_id),
            )
            self.assertIs(prior, prior_by_id[expected_id])
if __name__ == "__main__":
    unittest.main()