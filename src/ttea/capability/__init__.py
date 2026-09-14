"""Deterministic, domain-independent capability fixture definitions."""

from .catalogue import CAPABILITY_FIXTURES, capability_catalogue
from .models import (
    ActionRequirement, CAPABILITY_DIMENSIONS, EFFECT_KINDS,
    FIXTURE_SCHEMA_VERSION, PREDICATE_KINDS, CanonicalSubgoal,
    CapabilityAction, CapabilityFixture, CapabilityFixtureError, CapabilityInput,
    CapabilityOracle, OutputBudget, Predicate, PriorResult, StateEffect,
    canonical_fixture_bytes, fixture_digest, thaw_json,
)
from .evaluator import (
    CapabilityEvaluationResult, CapabilityVerdict, EvaluationState,
    EvaluationStatus, FailureCode, evaluate,
)

__all__ = [
    "ActionRequirement", "CAPABILITY_DIMENSIONS", "CAPABILITY_FIXTURES",
    "EFFECT_KINDS", "FIXTURE_SCHEMA_VERSION", "PREDICATE_KINDS",
    "CanonicalSubgoal", "CapabilityAction", "CapabilityFixture",
    "CapabilityFixtureError", "CapabilityInput", "CapabilityOracle",
    "OutputBudget", "Predicate", "PriorResult", "StateEffect",
    "canonical_fixture_bytes", "capability_catalogue", "fixture_digest",
    "thaw_json",
    "CapabilityEvaluationResult", "CapabilityVerdict", "EvaluationState",
    "EvaluationStatus", "FailureCode", "evaluate",
]
