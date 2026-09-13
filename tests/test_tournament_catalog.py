from __future__ import annotations

import unittest
from unittest.mock import patch

from ttea.benchmarking import OllamaClient, OwnedOllamaServer
from ttea.tournament.catalog import (
    CATALOG_BY_ID, CATALOG_SOURCE, CONTEXT_TARGET, INITIAL_CATALOG,
    REAL_CANDIDATES, REAL_CATALOG, RESEARCH_DATE, SAFETY_CATALOG_ENTRY,
    TIER_MEMBERSHIP, CandidateCatalogEntry, ResearchTier,
    build_real_candidate_manifest,
)
from ttea.tournament.models import TournamentResult
from ttea.tournament.scoring import select


EXPECTED_IDS = (
    "qwen35-08b",
    "qwen35-2b",
    "qwen35-4b",
    "qwen3-17b",
    "qwen3-4b-instruct",
    "granite4-3b",
    "granite4-3b-h",
    "llama32-3b",
    "phi4-mini",
    "ministral3-3b",
    "granite33-2b",
    "deepseek-r1-15b",
    "phi4-mini-reasoning",
)


class TournamentCatalogTests(unittest.TestCase):
    def test_expected_candidate_ids_and_order(self):
        self.assertEqual(tuple(candidate.candidate_id for candidate in REAL_CANDIDATES), EXPECTED_IDS)
        self.assertEqual(tuple(entry.candidate.candidate_id for entry in REAL_CATALOG), EXPECTED_IDS)

    def test_catalog_is_immutable_and_provenanced(self):
        self.assertIsInstance(INITIAL_CATALOG, tuple)
        self.assertEqual(SAFETY_CATALOG_ENTRY.research_date, "2026-09-13")
        self.assertEqual(RESEARCH_DATE, "2026-09-13")
        self.assertEqual(CATALOG_SOURCE, "official Ollama library")
        self.assertEqual(len(CATALOG_BY_ID), len(EXPECTED_IDS) + 1)
        with self.assertRaises(TypeError):
            CATALOG_BY_ID["extra"] = SAFETY_CATALOG_ENTRY

    def test_safety_control_remains_validation_only(self):
        self.assertTrue(SAFETY_CATALOG_ENTRY.candidate.validation_only)
        self.assertTrue(SAFETY_CATALOG_ENTRY.candidate.enabled)
        self.assertIsNone(SAFETY_CATALOG_ENTRY.published_package_bytes)

    def test_real_candidates_have_declared_runtime_independent_metadata(self):
        for entry in REAL_CATALOG:
            with self.subTest(candidate=entry.candidate.candidate_id):
                self.assertIsInstance(entry, CandidateCatalogEntry)
                self.assertTrue(entry.candidate.enabled)
                self.assertFalse(entry.candidate.validation_only)
                self.assertEqual(entry.candidate.variant.provider, "ollama")
                self.assertEqual(entry.candidate.context_target, CONTEXT_TARGET)
                self.assertTrue(entry.expected_quantization)
                self.assertEqual(entry.candidate.variant.quantization, entry.expected_quantization)
                self.assertGreater(entry.published_package_bytes, 0)
                self.assertEqual(entry.candidate.estimated_disk_bytes, entry.published_package_bytes)
                self.assertEqual(entry.research_date, RESEARCH_DATE)
                self.assertEqual(entry.source, CATALOG_SOURCE)

    def test_canonical_identities_are_unique(self):
        identities = [candidate.canonical_identity for candidate in REAL_CANDIDATES]
        self.assertEqual(len(identities), len(set(identities)))

    def test_tier_memberships_are_exact_and_valid(self):
        expected = {
            ResearchTier.LIGHT: {"qwen35-08b", "qwen3-17b", "granite33-2b", "deepseek-r1-15b"},
            ResearchTier.BALANCED: {"qwen35-2b", "granite4-3b", "granite4-3b-h", "llama32-3b"},
            ResearchTier.UPPER: {"qwen35-4b", "qwen3-4b-instruct", "phi4-mini", "ministral3-3b"},
            ResearchTier.SPECIALIST: {"phi4-mini-reasoning", "deepseek-r1-15b"},
        }
        self.assertEqual({tier: set(ids) for tier, ids in TIER_MEMBERSHIP.items()}, expected)
        self.assertTrue(set().union(*expected.values()) <= set(EXPECTED_IDS))

    def test_real_manifest_is_deterministic_and_has_no_rounds(self):
        first = build_real_candidate_manifest()
        second = build_real_candidate_manifest()
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.rounds, ())
        self.assertEqual(first.tasks, ())
        self.assertEqual(tuple(candidate.candidate_id for candidate in first.candidates), ("smollm2-safety",) + EXPECTED_IDS)

    def test_specialists_are_eligible_but_not_automatic_winners(self):
        regular = CATALOG_BY_ID["qwen35-2b"].candidate
        specialist = CATALOG_BY_ID["phi4-mini-reasoning"].candidate
        dimensions = {"compatibility": 0.8, "instruction_adherence": 0.8, "structured_output": 0.8, "planning": 0.8, "tool_use": 0.8, "reasoning": 0.8, "hallucination_resistance": 0.8, "stability": 0.8}
        lower = {**dimensions, "reasoning": 0.79}
        selections = select(
            (TournamentResult(regular.candidate_id, dimensions, {"peak_ram_bytes": 1}, True), TournamentResult(specialist.candidate_id, lower, {"peak_ram_bytes": 2}, True)),
            (regular, specialist),
        )
        self.assertEqual(selections[0].candidate_id, regular.candidate_id)
        self.assertNotEqual(specialist.candidate_id, "")

    def test_tags_are_official_style_without_cloud_or_community_namespaces(self):
        for entry in REAL_CATALOG:
            tag = entry.official_ollama_tag
            with self.subTest(tag=tag):
                self.assertNotIn("/", tag)
                self.assertNotIn("cloud", tag.lower())
                self.assertEqual(tag, entry.candidate.variant.model)

    def test_catalog_build_is_inert(self):
        with patch.object(OllamaClient, "pull", side_effect=AssertionError("pull called")) as pull, patch.object(OllamaClient, "generate", side_effect=AssertionError("generate called")) as generate, patch.object(OwnedOllamaServer, "start", side_effect=AssertionError("server start called")) as start, patch.object(OwnedOllamaServer, "shutdown", side_effect=AssertionError("server shutdown called")) as shutdown:
            manifest = build_real_candidate_manifest()
            self.assertEqual(len(manifest.candidates), 14)
            pull.assert_not_called()
            generate.assert_not_called()
            start.assert_not_called()
            shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
