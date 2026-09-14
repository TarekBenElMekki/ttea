"""Frozen, inert Capability Round 2 fixture catalogue."""

from __future__ import annotations

from .models import (
    ActionRequirement, CanonicalSubgoal, CapabilityAction, CapabilityFixture,
    CapabilityInput, CapabilityOracle, OutputBudget, Predicate, PriorResult,
    StateEffect,
)


_SINGLE_BUDGET = OutputBudget(512, 1)
_PLAN_BUDGET = OutputBudget(1200, 4)


def _fixture(
    fixture_id: str, dimensions: tuple[str, ...], input: CapabilityInput,
    subgoals: tuple[CanonicalSubgoal, ...], oracle: CapabilityOracle,
) -> CapabilityFixture:
    return CapabilityFixture(1, fixture_id, 1, dimensions, input, subgoals, oracle)


CAPABILITY_FIXTURES = (
    _fixture(
        "cap.protocol.basic", ("structured_action_protocol", "state_grounded_action"),
        CapabilityInput(
            "produce a prepared output", {"input_present": True, "prepared": False},
            actions=(CapabilityAction("prepare_output", (Predicate("state_equals", fact="input_present", value=True),
                                                           Predicate("state_equals", fact="prepared", value=False))),),
        ),
        (CanonicalSubgoal("prepare", ("prepare_output",)),),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("prepare",), output_budget=_SINGLE_BUDGET),
    ),
    _fixture(
        "cap.constraint.preserve_input", ("constraint_compliance", "structured_action_protocol"),
        CapabilityInput(
            "prepare output while retaining source input", {"input_present": True, "prepared": False},
            constraints=(Predicate("action_not_selected", action_id="discard_input"),),
            actions=(
                CapabilityAction("prepare_output", (Predicate("state_equals", fact="input_present", value=True),)),
                CapabilityAction("discard_input", (Predicate("state_exists", fact="input_present"),)),
            ),
        ),
        (CanonicalSubgoal("prepare", ("prepare_output",)),),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("prepare",), output_budget=_SINGLE_BUDGET),
    ),
    _fixture(
        "cap.policy.guard_boundary", ("constraint_compliance", "state_grounded_action"),
        CapabilityInput(
            "make item eligible for release", {"approved": False, "guard_enabled": True},
            policy_forbid=(Predicate("action_not_selected", action_id="override_guard"),),
            actions=(
                CapabilityAction("request_approval", (Predicate("state_equals", fact="approved", value=False),)),
                CapabilityAction("commit_output", (Predicate("state_equals", fact="approved", value=True),)),
                CapabilityAction("override_guard", (Predicate("state_equals", fact="guard_enabled", value=True),)),
            ),
        ),
        (CanonicalSubgoal("obtain_approval", ("request_approval",)),),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("obtain_approval",), output_budget=_SINGLE_BUDGET),
    ),
    _fixture(
        "cap.state.no_action", ("state_grounded_action", "structured_action_protocol"),
        CapabilityInput(
            "act only if an input is pending", {"input_present": False, "pending": False},
            actions=(
                CapabilityAction("inspect_input", (Predicate("state_equals", fact="input_present", value=True),)),
                CapabilityAction("wait", (Predicate("state_equals", fact="pending", value=True),)),
            ),
        ),
        (),
        CapabilityOracle(ActionRequirement.NO_ACTION, output_budget=_SINGLE_BUDGET),
    ),
    _fixture(
        "cap.state.precondition_choice", ("state_grounded_action", "constraint_compliance"),
        CapabilityInput(
            "finalize the validated item", {"validated": True, "expedite_token": False, "committed": False},
            actions=(
                CapabilityAction("fast_commit", (Predicate("state_equals", fact="validated", value=True),
                                                   Predicate("state_equals", fact="expedite_token", value=True))),
                CapabilityAction("commit_output", (Predicate("state_equals", fact="validated", value=True),
                                                     Predicate("state_equals", fact="committed", value=False))),
            ),
        ),
        (CanonicalSubgoal("commit", ("commit_output",)),),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("commit",), output_budget=_SINGLE_BUDGET),
    ),
    _fixture(
        "cap.plan.chain", ("mission_decomposition_planning", "state_grounded_action"),
        CapabilityInput(
            "create a committed output from supplied input",
            {"input_present": True, "inspected": False, "validated": False, "prepared": False, "committed": False},
            actions=(
                CapabilityAction("inspect_input", (Predicate("state_equals", fact="input_present", value=True),),
                                 (StateEffect("set_state", "inspected", True),)),
                CapabilityAction("validate_item", (Predicate("state_equals", fact="inspected", value=True),),
                                 (StateEffect("set_state", "validated", True),)),
                CapabilityAction("prepare_output", (Predicate("state_equals", fact="validated", value=True),),
                                 (StateEffect("set_state", "prepared", True),)),
                CapabilityAction("commit_output", (Predicate("state_equals", fact="prepared", value=True),),
                                 (StateEffect("set_state", "committed", True),)),
            ),
        ),
        (
            CanonicalSubgoal("collect", ("inspect_input",)),
            CanonicalSubgoal("validate", ("validate_item",)),
            CanonicalSubgoal("prepare", ("prepare_output",)),
            CanonicalSubgoal("commit", ("commit_output",)),
        ),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("collect", "validate", "prepare", "commit"),
                         (("collect", "validate"), ("validate", "prepare"), ("prepare", "commit")),
                         (("collect", "validate"), ("validate", "prepare"), ("prepare", "commit")),
                         output_budget=_PLAN_BUDGET),
    ),
    _fixture(
        "cap.plan.fork_join", ("mission_decomposition_planning", "structured_action_protocol"),
        CapabilityInput(
            "prepare an output after independent input and policy checks",
            {"input_available": True, "policy_available": True, "input_checked": False,
             "policy_checked": False, "validated": False, "prepared": False},
            actions=(
                CapabilityAction("inspect_input", (Predicate("state_equals", fact="input_available", value=True),),
                                 (StateEffect("set_state", "input_checked", True),)),
                CapabilityAction("inspect_policy", (Predicate("state_equals", fact="policy_available", value=True),),
                                 (StateEffect("set_state", "policy_checked", True),)),
                CapabilityAction("validate_item", (Predicate("state_equals", fact="input_checked", value=True),
                                                    Predicate("state_equals", fact="policy_checked", value=True)),
                                 (StateEffect("set_state", "validated", True),)),
                CapabilityAction("prepare_output", (Predicate("state_equals", fact="validated", value=True),),
                                 (StateEffect("set_state", "prepared", True),)),
            ),
        ),
        (
            CanonicalSubgoal("collect_input", ("inspect_input",)),
            CanonicalSubgoal("collect_policy", ("inspect_policy",)),
            CanonicalSubgoal("validate", ("validate_item",)),
            CanonicalSubgoal("prepare", ("prepare_output",)),
        ),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED,
                         ("collect_input", "collect_policy", "validate", "prepare"),
                         (("collect_input", "validate"), ("collect_policy", "validate"), ("validate", "prepare")),
                         (("collect_input", "validate"), ("collect_policy", "validate"), ("validate", "prepare")),
                         output_budget=_PLAN_BUDGET),
    ),
    _fixture(
        "cap.replan.failed_alternative", ("adaptive_replanning_recovery", "state_grounded_action"),
        CapabilityInput(
            "prepare output despite a failed primary route",
            {"primary_available": False, "alternative_available": True, "prepared": False},
            actions=(
                CapabilityAction("prepare_primary", (Predicate("state_equals", fact="primary_available", value=True),)),
                CapabilityAction("use_alternative", (Predicate("state_equals", fact="alternative_available", value=True),
                                                       Predicate("state_equals", fact="prepared", value=False))),
            ),
            prior_result=PriorResult("prepare_primary", "unavailable"),
        ),
        (CanonicalSubgoal("prepare", ("use_alternative",)),),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("prepare",),
                         replan_rules=(Predicate("action_not_repeated", action_id="prepare_primary"),),
                         output_budget=_SINGLE_BUDGET),
    ),
    _fixture(
        "cap.replan.state_delta", ("adaptive_replanning_recovery", "mission_decomposition_planning"),
        CapabilityInput(
            "complete delivery after a changed state", {"prepared": False, "committed": False},
            actions=(
                CapabilityAction("prepare_output", (Predicate("state_equals", fact="prepared", value=False),)),
                CapabilityAction("commit_output", (Predicate("state_equals", fact="prepared", value=True),
                                                     Predicate("state_equals", fact="committed", value=False))),
            ),
            prior_result=PriorResult("prepare_output", "completed"),
            state_delta={"prepared": True},
        ),
        (CanonicalSubgoal("commit", ("commit_output",)),),
        CapabilityOracle(ActionRequirement.ACTION_REQUIRED, ("commit",),
                         replan_rules=(Predicate("action_not_repeated", action_id="prepare_output"),),
                         output_budget=_SINGLE_BUDGET),
    ),
)


def capability_catalogue() -> tuple[CapabilityFixture, ...]:
    """Return the frozen fixture sequence without runtime side effects."""
    return CAPABILITY_FIXTURES
