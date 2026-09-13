"""Declarative September 2026 local-model candidate catalog.

This module contains published library metadata and provisional planning
estimates only. It never discovers, provisions, or executes a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .manifest import SAFETY_CANDIDATE, TournamentManifest
from .models import CandidateVariant, TournamentCandidate, TournamentPhase, TournamentRound, TournamentTask


GIB = 1024 * 1024 * 1024
RESEARCH_DATE = "2026-09-13"
CATALOG_SOURCE = "official Ollama library"
CONTEXT_TARGET = 4096
SURVIVAL_PROMPT = "Reply exactly with: TTEA_SURVIVAL_OK"
SURVIVAL_TASK = TournamentTask(
    "survival_probe", "hardware_survival", SURVIVAL_PROMPT,
    "bounded non-empty text", "runtime_terminal", 1.0,
    CONTEXT_TARGET, 32, 180.0,
)
SURVIVAL_ROUND = TournamentRound(
    "survival_4k", TournamentPhase.SURVIVAL, (SURVIVAL_TASK.task_id,),
    1, CONTEXT_TARGET, 32, "fresh_resource_admission",
    "benchmark_supervisor_request_300_model_180",
    "pass_fail_hardware_survival", "hard_safety_gates",
)


class ResearchTier(str, Enum):
    LIGHT = "LIGHT"
    BALANCED = "BALANCED"
    UPPER = "UPPER"
    SPECIALIST = "SPECIALIST"


@dataclass(frozen=True)
class ProvisionalResourceEstimate:
    """Conservative planning capacity; not a measured runtime observation."""

    ram_bytes: int
    vram_bytes: int

    def __post_init__(self) -> None:
        for name in ("ram_bytes", "vram_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


TIER_RESOURCE_ESTIMATES: Mapping[ResearchTier, ProvisionalResourceEstimate] = MappingProxyType({
    ResearchTier.LIGHT: ProvisionalResourceEstimate(6 * GIB, 4 * GIB),
    ResearchTier.BALANCED: ProvisionalResourceEstimate(10 * GIB, 6 * GIB),
    ResearchTier.UPPER: ProvisionalResourceEstimate(14 * GIB, 8 * GIB),
    ResearchTier.SPECIALIST: ProvisionalResourceEstimate(14 * GIB, 8 * GIB),
})


@dataclass(frozen=True)
class CandidateCatalogEntry:
    candidate: TournamentCandidate
    tier: ResearchTier
    research_role: str
    official_ollama_tag: str
    published_package_bytes: int | None
    expected_quantization: str
    parameter_class: str
    capabilities_notes: tuple[str, ...]
    research_date: str = RESEARCH_DATE
    source: str = CATALOG_SOURCE

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TournamentCandidate):
            raise ValueError("candidate must be a TournamentCandidate")
        if not isinstance(self.tier, ResearchTier):
            raise ValueError("tier must be a ResearchTier")
        for name in ("research_role", "official_ollama_tag", "expected_quantization", "parameter_class", "research_date", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.official_ollama_tag != self.candidate.variant.model:
            raise ValueError("official_ollama_tag must match the candidate model")
        if self.candidate.variant.provider != "ollama":
            raise ValueError("initial catalog candidates must use the ollama provider")
        if self.candidate.variant.quantization != self.expected_quantization:
            raise ValueError("expected_quantization must match candidate quantization")
        if self.published_package_bytes is not None and (
            not isinstance(self.published_package_bytes, int)
            or isinstance(self.published_package_bytes, bool)
            or self.published_package_bytes <= 0
        ):
            raise ValueError("published_package_bytes must be a positive integer when present")
        if self.candidate.estimated_disk_bytes != self.published_package_bytes:
            raise ValueError("candidate disk estimate must use published package bytes")
        if not isinstance(self.capabilities_notes, tuple) or not self.capabilities_notes or any(not isinstance(note, str) or not note.strip() for note in self.capabilities_notes):
            raise ValueError("capabilities_notes must be a non-empty tuple of non-empty strings")


def _entry(
    candidate_id: str,
    model: str,
    family: str,
    parameter_size: str,
    quantization: str,
    package_gb: float,
    purpose: str,
    tier: ResearchTier,
    *notes: str,
) -> CandidateCatalogEntry:
    estimate = TIER_RESOURCE_ESTIMATES[tier]
    package_bytes = round(package_gb * 1_000_000_000)
    candidate = TournamentCandidate(
        candidate_id=candidate_id,
        family=family,
        variant=CandidateVariant("ollama", model, quantization, parameter_size),
        estimated_disk_bytes=package_bytes,
        estimated_ram_bytes=estimate.ram_bytes,
        estimated_vram_bytes=estimate.vram_bytes,
        context_target=CONTEXT_TARGET,
        purpose=purpose,
        enabled=True,
        validation_only=False,
    )
    return CandidateCatalogEntry(
        candidate=candidate,
        tier=tier,
        research_role=purpose,
        official_ollama_tag=model,
        published_package_bytes=package_bytes,
        expected_quantization=quantization,
        parameter_class=parameter_size,
        capabilities_notes=notes or ("candidate requires future tournament evidence",),
    )


REAL_CATALOG: tuple[CandidateCatalogEntry, ...] = (
    _entry("qwen35-08b", "qwen3.5:0.8b", "qwen3.5", "0.8B", "Q4_K_M", 1.0, "ultra-light agent candidate", ResearchTier.LIGHT, "ultra-light agent research"),
    _entry("qwen35-2b", "qwen3.5:2b", "qwen3.5", "2B class", "Q4_K_M", 2.7, "balanced agent candidate", ResearchTier.BALANCED, "balanced agent research"),
    _entry("qwen35-4b", "qwen3.5:4b-q4_K_M", "qwen3.5", "4.66B", "Q4_K_M", 3.4, "upper-end primary candidate", ResearchTier.UPPER, "upper capacity candidate"),
    _entry("qwen3-17b", "qwen3:1.7b-q4_K_M", "qwen3", "1.7B", "Q4_K_M", 1.4, "lightweight previous-generation control", ResearchTier.LIGHT, "previous-generation control"),
    _entry("qwen3-4b-instruct", "qwen3:4b-instruct-2507-q4_K_M", "qwen3", "4.02B", "Q4_K_M", 2.5, "strong instruction/tool baseline", ResearchTier.UPPER, "instruction and tool baseline"),
    _entry("granite4-3b", "granite4:3b", "granite4", "3.4B", "Q4_K_M", 2.1, "tool-oriented primary candidate", ResearchTier.BALANCED, "tool-oriented research"),
    _entry("granite4-3b-h", "granite4:micro-h", "granite4-hybrid", "3.19B", "Q4_K_M", 1.9, "hybrid architecture efficiency candidate", ResearchTier.BALANCED, "hybrid architecture efficiency"),
    _entry("llama32-3b", "llama3.2:3b-instruct-q4_K_M", "llama3.2", "3.21B", "Q4_K_M", 2.0, "mature agent/tool baseline", ResearchTier.BALANCED, "mature agent and tool baseline"),
    _entry("phi4-mini", "phi4-mini:3.8b-q4_K_M", "phi4-mini", "3.84B", "Q4_K_M", 2.5, "reasoning and function-calling candidate", ResearchTier.UPPER, "reasoning and function calling"),
    _entry("ministral3-3b", "ministral-3:3b", "ministral-3", "3.85B", "Q4_K_M", 3.0, "agentic/function-calling candidate", ResearchTier.UPPER, "agentic and function calling"),
    _entry("granite33-2b", "granite3.3:2b", "granite3.3", "2.53B", "Q4_K_M", 1.5, "small tool/instruction baseline", ResearchTier.LIGHT, "small tool and instruction baseline"),
    _entry("deepseek-r1-15b", "deepseek-r1:1.5b", "deepseek-r1", "1.78B", "Q4_K_M", 1.1, "small reasoning specialist candidate", ResearchTier.LIGHT, "reasoning specialist"),
    _entry("phi4-mini-reasoning", "phi4-mini-reasoning:3.8b-q4_K_M", "phi4-mini-reasoning", "3.8B", "Q4_K_M", 3.2, "reasoning specialist candidate", ResearchTier.SPECIALIST, "reasoning specialist"),
)

REAL_CANDIDATES: tuple[TournamentCandidate, ...] = tuple(entry.candidate for entry in REAL_CATALOG)
SAFETY_CATALOG_ENTRY = CandidateCatalogEntry(
    candidate=SAFETY_CANDIDATE,
    tier=ResearchTier.LIGHT,
    research_role="safety validation control",
    official_ollama_tag=SAFETY_CANDIDATE.variant.model,
    published_package_bytes=None,
    expected_quantization=SAFETY_CANDIDATE.variant.quantization or "Q4_0",
    parameter_class=SAFETY_CANDIDATE.variant.parameter_size or "135M",
    capabilities_notes=("validation-only safety control",),
)
INITIAL_CATALOG: tuple[CandidateCatalogEntry, ...] = (SAFETY_CATALOG_ENTRY,) + REAL_CATALOG
CATALOG_BY_ID: Mapping[str, CandidateCatalogEntry] = MappingProxyType({entry.candidate.candidate_id: entry for entry in INITIAL_CATALOG})
TIER_MEMBERSHIP: Mapping[ResearchTier, tuple[str, ...]] = MappingProxyType({
    ResearchTier.LIGHT: ("qwen35-08b", "qwen3-17b", "granite33-2b", "deepseek-r1-15b"),
    ResearchTier.BALANCED: ("qwen35-2b", "granite4-3b", "granite4-3b-h", "llama32-3b"),
    ResearchTier.UPPER: ("qwen35-4b", "qwen3-4b-instruct", "phi4-mini", "ministral3-3b"),
    ResearchTier.SPECIALIST: ("phi4-mini-reasoning", "deepseek-r1-15b"),
})


def build_real_candidate_manifest() -> TournamentManifest:
    """Return the closed candidate manifest; this does not start a tournament."""
    return TournamentManifest(candidates=(SAFETY_CANDIDATE,) + REAL_CANDIDATES)


def build_survival_manifest() -> TournamentManifest:
    """Return the closed executable first-round manifest in catalog order."""
    return TournamentManifest(REAL_CANDIDATES, (SURVIVAL_ROUND,), (SURVIVAL_TASK,))
