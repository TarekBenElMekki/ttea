from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from ttea.resources import ResourceRequest
from ttea.tournament.capability import (
    CapabilityStoreError,
    ROUND2_CANDIDATES,
    ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST,
    ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID,
    load_round2_manifest,
    round2_manifest_from_source,
)
from ttea.tournament.catalog import build_survival_manifest
from ttea.tournament.manifest import ManifestMismatchError, TournamentManifest
from ttea.tournament.models import CandidateVariant, TournamentCandidate
from ttea.tournament.ollama_provider import (
    OllamaTournamentProvider,
    ProvisioningRejected,
)
from ttea.tournament.runner import TournamentExecutionRequest


def execution_request(candidate: TournamentCandidate, *, round_id: str = "capability_4k"):
    return TournamentExecutionRequest(
        candidate.candidate_id,
        round_id,
        "cap.synthetic",
        0,
        "ollama",
        candidate.variant.model,
        '{"mode":"NO_ACTION","plan":[]}',
        4096,
        32,
        180,
        0,
        4,
        0,
    )


def resource_request(candidate: TournamentCandidate):
    return ResourceRequest(
        workload_type=f"tournament-capability:{candidate.candidate_id}",
        estimated_ram_bytes=1,
        estimated_disk_bytes=0,
        estimated_gpu_vram_bytes=1,
    )


class CapabilityProviderBridgeTests(unittest.TestCase):
    def test_authoritative_survival_manifest_matches_frozen_digest(self):
        source = build_survival_manifest()
        self.assertEqual(source.digest, ROUND2_SOURCE_SURVIVAL_MANIFEST_DIGEST)

    def test_round2_manifest_reuses_exact_source_candidate_objects_in_frozen_order(self):
        source = build_survival_manifest()
        derived = round2_manifest_from_source(source)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in derived.candidates),
            ROUND2_CANDIDATES,
        )
        self.assertEqual(derived.rounds, ())
        self.assertEqual(derived.tasks, ())
        for candidate_id, candidate in zip(ROUND2_CANDIDATES, derived.candidates):
            self.assertIs(candidate, source.candidate(candidate_id))
        self.assertEqual(
            derived.candidate("qwen35-08b").variant.quantization,
            "Q8_0",
        )

    def test_round2_manifest_rejects_wrong_source_digest(self):
        source = build_survival_manifest()
        with self.assertRaises(ManifestMismatchError):
            round2_manifest_from_source(source, expected_digest="0" * 64)

    def test_round2_manifest_rejects_disabled_survivor(self):
        source = build_survival_manifest()
        candidates = list(source.candidates)
        index = next(
            i for i, item in enumerate(candidates)
            if item.candidate_id == ROUND2_CANDIDATES[0]
        )
        candidates[index] = replace(candidates[index], enabled=False)
        altered = TournamentManifest(
            candidates=tuple(candidates),
            rounds=source.rounds,
            tasks=source.tasks,
        )
        with self.assertRaises(CapabilityStoreError):
            round2_manifest_from_source(
                altered,
                expected_digest=altered.digest,
            )

    def test_round2_manifest_loader_uses_authoritative_campaign_path(self):
        source = build_survival_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / ROUND2_SOURCE_SURVIVAL_TOURNAMENT_ID
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(
                json.dumps(source.to_dict()),
                encoding="utf-8",
            )
            derived = load_round2_manifest(root)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in derived.candidates),
            ROUND2_CANDIDATES,
        )

    def test_generic_execution_rejects_candidate_outside_provider_manifest_before_runtime(self):
        source = build_survival_manifest()
        candidate = source.candidate(ROUND2_CANDIDATES[0])
        provider = object.__new__(OllamaTournamentProvider)
        provider._allowed = {
            candidate.candidate_id: candidate.variant.model,
        }
        arbitrary = TournamentCandidate(
            "arbitrary",
            "x",
            CandidateVariant("ollama", "evil:latest", "Q4_K_M", "1B"),
            1,
            1,
            1,
            4096,
            "not persisted",
        )
        with self.assertRaises(ProvisioningRejected):
            provider.execute_prompt(
                arbitrary,
                execution_request(arbitrary),
                resource_request(arbitrary),
            )

    def test_generic_execution_has_no_survival_round_gate(self):
        class RuntimeReached(RuntimeError):
            pass

        source = build_survival_manifest()
        candidate = source.candidate(ROUND2_CANDIDATES[0])
        provider = object.__new__(OllamaTournamentProvider)
        provider._allowed = {
            candidate.candidate_id: candidate.variant.model,
        }
        provider.default_identity = Mock(side_effect=RuntimeReached("runtime reached"))

        with self.assertRaisesRegex(RuntimeReached, "runtime reached"):
            provider.execute_prompt(
                candidate,
                execution_request(candidate, round_id="capability_4k"),
                resource_request(candidate),
            )

    def test_execute_survival_is_thin_backward_compatible_wrapper(self):
        source = build_survival_manifest()
        candidate = source.candidate(ROUND2_CANDIDATES[0])
        request = execution_request(candidate, round_id="survival_4k")
        resources = resource_request(candidate)
        expected = object()

        provider = object.__new__(OllamaTournamentProvider)
        provider.execute_prompt = Mock(return_value=expected)

        result = OllamaTournamentProvider.execute_survival(
            provider,
            candidate,
            request,
            resources,
        )

        self.assertIs(result, expected)
        provider.execute_prompt.assert_called_once_with(
            candidate,
            request,
            resources,
        )


if __name__ == "__main__":
    unittest.main()