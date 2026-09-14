"""Focused regression tests for immutable Capability Round 2 fixtures."""

from __future__ import annotations

from dataclasses import replace
import unittest

from ttea.capability import (
    ActionRequirement, CAPABILITY_FIXTURES, CanonicalSubgoal, CapabilityAction,
    CapabilityFixture, CapabilityFixtureError, CapabilityInput, CapabilityOracle,
    OutputBudget, Predicate, StateEffect, canonical_fixture_bytes,
    capability_catalogue, fixture_digest,
)


EXPECTED_IDS = (
    "cap.protocol.basic",
    "cap.constraint.preserve_input",
    "cap.policy.guard_boundary",
    "cap.state.no_action",
    "cap.state.precondition_choice",
    "cap.plan.chain",
    "cap.plan.fork_join",
    "cap.replan.failed_alternative",
    "cap.replan.state_delta",
)


def _fixture(*, actions: tuple[CapabilityAction, ...] | None = None,
             subgoals: tuple[CanonicalSubgoal, ...] | None = None,
             oracle: CapabilityOracle | None = None,
             state: dict[str, object] | None = None) -> CapabilityFixture:
    action_items = actions or (CapabilityAction("prepare", (Predicate("state_equals", fact="ready", value=True),)),)
    subgoal_items = subgoals or (CanonicalSubgoal("prepare", ("prepare",)),)
    oracle_item = oracle or CapabilityOracle(
        ActionRequirement.ACTION_REQUIRED, ("prepare",), output_budget=OutputBudget(512, 1),
    )
    return CapabilityFixture(
        1, "cap.test.fixture", 1, ("structured_action_protocol",),
        CapabilityInput("prepare", state or {"ready": True}, actions=action_items),
        subgoal_items, oracle_item,
    )


class CapabilityFixtureTests(unittest.TestCase):
    def test_catalogue_has_frozen_unique_order(self) -> None:
        self.assertEqual(tuple(item.id for item in CAPABILITY_FIXTURES), EXPECTED_IDS)
        self.assertEqual(len({item.id for item in CAPABILITY_FIXTURES}), len(EXPECTED_IDS))

    def test_catalogue_construction_is_inert(self) -> None:
        self.assertIs(capability_catalogue(), CAPABILITY_FIXTURES)

    def test_models_and_nested_data_are_immutable_and_defensively_copied(self) -> None:
        source = {"ready": True, "nested": {"value": "original"}}
        fixture = _fixture(state=source)
        source["nested"]["value"] = "changed"
        with self.assertRaises(TypeError):
            fixture.input.state["ready"] = False
        with self.assertRaises(TypeError):
            fixture.input.state["nested"]["value"] = "changed again"
        with self.assertRaises(AttributeError):
            fixture.id = "cap.changed"
        self.assertEqual(fixture.input.state["nested"]["value"], "original")

    def test_canonical_serialization_and_digest_are_deterministic(self) -> None:
        first = _fixture(state={"ready": True, "other": False})
        second = _fixture(state={"other": False, "ready": True})
        self.assertEqual(canonical_fixture_bytes(first), canonical_fixture_bytes(second))
        self.assertEqual(fixture_digest(first), fixture_digest(second))
        changed = replace(first, input=replace(first.input, mission="different mission"))
        self.assertNotEqual(fixture_digest(first), fixture_digest(changed))

    def test_invalid_schema_and_unknown_vocabulary_are_rejected(self) -> None:
        with self.assertRaises(CapabilityFixtureError):
            replace(_fixture(), schema_version=2)
        with self.assertRaises(CapabilityFixtureError):
            Predicate("arbitrary_expression")
        with self.assertRaises(CapabilityFixtureError):
            StateEffect("remove_state", "ready", True)

    def test_unknown_action_and_subgoal_references_are_rejected(self) -> None:
        with self.assertRaises(CapabilityFixtureError):
            _fixture(subgoals=(CanonicalSubgoal("prepare", ("missing",)),))
        with self.assertRaises(CapabilityFixtureError):
            _fixture(oracle=CapabilityOracle(
                ActionRequirement.ACTION_REQUIRED, ("missing",), output_budget=OutputBudget(512, 1),
            ))

    def test_cycle_and_duplicate_action_or_subgoal_are_rejected(self) -> None:
        with self.assertRaises(CapabilityFixtureError):
            _fixture(oracle=CapabilityOracle(
                ActionRequirement.ACTION_REQUIRED, ("prepare",), (("prepare", "prepare"),),
                output_budget=OutputBudget(512, 1),
            ))
        action = CapabilityAction("prepare")
        with self.assertRaises(CapabilityFixtureError):
            _fixture(actions=(action, action))
        subgoal = CanonicalSubgoal("prepare", ("prepare",))
        with self.assertRaises(CapabilityFixtureError):
            _fixture(subgoals=(subgoal, subgoal))

    def test_invalid_budget_and_state_effect_references_are_rejected(self) -> None:
        with self.assertRaises(CapabilityFixtureError):
            OutputBudget(-1, 1)
        with self.assertRaises(CapabilityFixtureError):
            _fixture(actions=(CapabilityAction("prepare", effects=(StateEffect("set_state", "unknown", True),)),))
        with self.assertRaises(CapabilityFixtureError):
            _fixture(actions=(CapabilityAction(
                "prepare", effects=(StateEffect("set_state", "ready", True), StateEffect("set_state", "ready", False)),
            ),))

    def test_frozen_catalogue_and_effectful_plans_validate(self) -> None:
        self.assertTrue(all(isinstance(item, CapabilityFixture) for item in CAPABILITY_FIXTURES))
        chain = next(item for item in CAPABILITY_FIXTURES if item.id == "cap.plan.chain")
        self.assertEqual(chain.oracle.required_subgoal_ids, ("collect", "validate", "prepare", "commit"))
        self.assertTrue(any(action.effects for action in chain.input.actions))

    def test_fork_join_has_only_declared_converging_dependencies(self) -> None:
        fixture = next(item for item in CAPABILITY_FIXTURES if item.id == "cap.plan.fork_join")
        self.assertNotIn(("collect_input", "collect_policy"), fixture.oracle.dependency_edges)
        self.assertNotIn(("collect_policy", "collect_input"), fixture.oracle.dependency_edges)
        self.assertEqual(
            fixture.oracle.dependency_edges,
            (("collect_input", "validate"), ("collect_policy", "validate"), ("validate", "prepare")),
        )


if __name__ == "__main__":
    unittest.main()
