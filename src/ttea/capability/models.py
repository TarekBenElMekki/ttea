"""Immutable, closed-world Capability Round 2 fixture models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


FIXTURE_SCHEMA_VERSION = 1
CAPABILITY_DIMENSIONS = (
    "constraint_compliance",
    "structured_action_protocol",
    "state_grounded_action",
    "mission_decomposition_planning",
    "adaptive_replanning_recovery",
    "actionable_efficiency",
)

PREDICATE_KINDS = frozenset((
    "state_equals", "state_exists", "state_absent", "action_selected",
    "action_not_selected", "action_precondition", "action_attempted",
    "action_failed", "action_not_repeated", "subgoal_present",
    "no_action_selected",
))
EFFECT_KINDS = frozenset(("set_state",))


class CapabilityFixtureError(ValueError):
    """Raised when authored capability fixture data is not closed and valid."""


class ActionRequirement(str, Enum):
    ACTION_REQUIRED = "ACTION_REQUIRED"
    NO_ACTION = "NO_ACTION"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityFixtureError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CapabilityFixtureError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CapabilityFixtureError(f"{name} must be a non-negative integer")
    return value


def _freeze_json(value: Any, name: str = "value") -> Any:
    """Defensively freeze JSON-compatible data and reject non-determinism."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapabilityFixtureError(f"{name} must not contain a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CapabilityFixtureError(f"{name} mapping keys must be strings")
            copied[key] = _freeze_json(item, name)
        return MappingProxyType(copied)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, name) for item in value)
    raise CapabilityFixtureError(f"{name} must be JSON-compatible")


def thaw_json(value: Any) -> Any:
    """Return a mutable JSON-compatible copy for canonical serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class Predicate:
    kind: str
    fact: str | None = None
    value: Any = None
    action_id: str | None = None
    subgoal_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in PREDICATE_KINDS:
            raise CapabilityFixtureError(f"unknown predicate kind: {self.kind!r}")
        if self.kind == "state_equals":
            _non_empty(self.fact, "predicate fact")
            object.__setattr__(self, "value", _freeze_json(self.value, "predicate value"))
            if self.action_id is not None or self.subgoal_id is not None:
                raise CapabilityFixtureError("state_equals accepts only fact and value")
        elif self.kind in {"state_exists", "state_absent"}:
            _non_empty(self.fact, "predicate fact")
            if self.value is not None or self.action_id is not None or self.subgoal_id is not None:
                raise CapabilityFixtureError(f"{self.kind} accepts only fact")
        elif self.kind in {
            "action_selected", "action_not_selected", "action_precondition",
            "action_attempted", "action_failed", "action_not_repeated",
        }:
            _non_empty(self.action_id, "predicate action_id")
            if self.fact is not None or self.value is not None or self.subgoal_id is not None:
                raise CapabilityFixtureError(f"{self.kind} accepts only action_id")
        elif self.kind == "subgoal_present":
            _non_empty(self.subgoal_id, "predicate subgoal_id")
            if self.fact is not None or self.value is not None or self.action_id is not None:
                raise CapabilityFixtureError("subgoal_present accepts only subgoal_id")
        else:  # no_action_selected
            if any(value is not None for value in (self.fact, self.value, self.action_id, self.subgoal_id)):
                raise CapabilityFixtureError("no_action_selected accepts no arguments")

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind}
        if self.fact is not None:
            result["fact"] = self.fact
        if self.kind == "state_equals":
            result["value"] = thaw_json(self.value)
        if self.action_id is not None:
            result["action_id"] = self.action_id
        if self.subgoal_id is not None:
            result["subgoal_id"] = self.subgoal_id
        return result


@dataclass(frozen=True)
class StateEffect:
    kind: str
    fact: str
    value: Any

    def __post_init__(self) -> None:
        if self.kind not in EFFECT_KINDS:
            raise CapabilityFixtureError(f"unknown effect kind: {self.kind!r}")
        _non_empty(self.fact, "effect fact")
        object.__setattr__(self, "value", _freeze_json(self.value, "effect value"))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "fact": self.fact, "value": thaw_json(self.value)}


@dataclass(frozen=True)
class CapabilityAction:
    id: str
    preconditions: tuple[Predicate, ...] = ()
    effects: tuple[StateEffect, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.id, "action id")
        preconditions = tuple(self.preconditions)
        effects = tuple(self.effects)
        if any(not isinstance(item, Predicate) for item in preconditions):
            raise CapabilityFixtureError("action preconditions must be predicates")
        if any(not isinstance(item, StateEffect) for item in effects):
            raise CapabilityFixtureError("action effects must be state effects")
        targets = [item.fact for item in effects]
        if len(targets) != len(set(targets)):
            raise CapabilityFixtureError("action may not set the same state fact more than once")
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "effects", effects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "preconditions": [item.to_dict() for item in self.preconditions],
            "effects": [item.to_dict() for item in self.effects],
        }


@dataclass(frozen=True)
class CanonicalSubgoal:
    id: str
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty(self.id, "subgoal id")
        action_ids = tuple(self.action_ids)
        if not action_ids or any(not isinstance(item, str) or not item.strip() for item in action_ids):
            raise CapabilityFixtureError("subgoal action_ids must be a non-empty tuple of IDs")
        if len(action_ids) != len(set(action_ids)):
            raise CapabilityFixtureError("subgoal action_ids must be unique")
        object.__setattr__(self, "action_ids", action_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "action_ids": list(self.action_ids)}


@dataclass(frozen=True)
class PriorResult:
    action_id: str
    outcome: str

    def __post_init__(self) -> None:
        _non_empty(self.action_id, "prior result action_id")
        _non_empty(self.outcome, "prior result outcome")

    def to_dict(self) -> dict[str, str]:
        return {"action_id": self.action_id, "outcome": self.outcome}


@dataclass(frozen=True)
class CapabilityInput:
    mission: str
    state: Mapping[str, Any]
    constraints: tuple[Predicate, ...] = ()
    policy_forbid: tuple[Predicate, ...] = ()
    policy_require: tuple[Predicate, ...] = ()
    actions: tuple[CapabilityAction, ...] = ()
    prior_result: PriorResult | None = None
    state_delta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _non_empty(self.mission, "mission")
        state = _freeze_json(self.state, "state")
        if not isinstance(state, MappingProxyType):
            raise CapabilityFixtureError("state must be an object")
        state_delta = None if self.state_delta is None else _freeze_json(self.state_delta, "state_delta")
        if state_delta is not None and not isinstance(state_delta, MappingProxyType):
            raise CapabilityFixtureError("state_delta must be an object")
        for name in ("constraints", "policy_forbid", "policy_require"):
            value = tuple(getattr(self, name))
            if any(not isinstance(item, Predicate) for item in value):
                raise CapabilityFixtureError(f"{name} must contain predicates")
            object.__setattr__(self, name, value)
        actions = tuple(self.actions)
        if any(not isinstance(item, CapabilityAction) for item in actions):
            raise CapabilityFixtureError("actions must contain capability actions")
        if self.prior_result is not None and not isinstance(self.prior_result, PriorResult):
            raise CapabilityFixtureError("prior_result must be PriorResult or None")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "state_delta", state_delta)
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission": self.mission,
            "state": thaw_json(self.state),
            "constraints": [item.to_dict() for item in self.constraints],
            "policy_forbid": [item.to_dict() for item in self.policy_forbid],
            "policy_require": [item.to_dict() for item in self.policy_require],
            "actions": [item.to_dict() for item in self.actions],
            "prior_result": None if self.prior_result is None else self.prior_result.to_dict(),
            "state_delta": None if self.state_delta is None else thaw_json(self.state_delta),
        }


@dataclass(frozen=True)
class OutputBudget:
    max_bytes: int
    max_steps: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.max_bytes, "max_bytes")
        _nonnegative_int(self.max_steps, "max_steps")

    def to_dict(self) -> dict[str, int]:
        return {"max_bytes": self.max_bytes, "max_steps": self.max_steps}


@dataclass(frozen=True)
class CapabilityOracle:
    action_requirement: ActionRequirement
    required_subgoal_ids: tuple[str, ...] = ()
    dependency_edges: tuple[tuple[str, str], ...] = ()
    required_order: tuple[tuple[str, str], ...] = ()
    replan_rules: tuple[Predicate, ...] = ()
    output_budget: OutputBudget = OutputBudget(512, 1)

    def __post_init__(self) -> None:
        if not isinstance(self.action_requirement, ActionRequirement):
            raise CapabilityFixtureError("action_requirement must be an ActionRequirement")
        subgoals = tuple(self.required_subgoal_ids)
        if len(subgoals) != len(set(subgoals)) or any(not isinstance(item, str) or not item.strip() for item in subgoals):
            raise CapabilityFixtureError("required_subgoal_ids must be unique non-empty IDs")
        edges = _pairs(self.dependency_edges, "dependency_edges")
        ordering = _pairs(self.required_order, "required_order")
        replan_rules = tuple(self.replan_rules)
        if any(not isinstance(item, Predicate) for item in replan_rules):
            raise CapabilityFixtureError("replan_rules must contain predicates")
        if not isinstance(self.output_budget, OutputBudget):
            raise CapabilityFixtureError("output_budget must be OutputBudget")
        if self.action_requirement is ActionRequirement.NO_ACTION and (subgoals or edges or ordering):
            raise CapabilityFixtureError("NO_ACTION fixtures may not require a plan")
        if self.action_requirement is ActionRequirement.ACTION_REQUIRED and not subgoals:
            raise CapabilityFixtureError("ACTION_REQUIRED fixtures require canonical subgoals")
        object.__setattr__(self, "required_subgoal_ids", subgoals)
        object.__setattr__(self, "dependency_edges", edges)
        object.__setattr__(self, "required_order", ordering)
        object.__setattr__(self, "replan_rules", replan_rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_requirement": self.action_requirement.value,
            "required_subgoal_ids": list(self.required_subgoal_ids),
            "dependency_edges": [list(item) for item in self.dependency_edges],
            "required_order": [list(item) for item in self.required_order],
            "replan_rules": [item.to_dict() for item in self.replan_rules],
            "output_budget": self.output_budget.to_dict(),
        }


def _pairs(value: Sequence[Sequence[str]], name: str) -> tuple[tuple[str, str], ...]:
    pairs = tuple(tuple(item) for item in value)
    if any(len(item) != 2 or any(not isinstance(part, str) or not part.strip() for part in item) for item in pairs):
        raise CapabilityFixtureError(f"{name} must contain pairs of non-empty IDs")
    if len(pairs) != len(set(pairs)):
        raise CapabilityFixtureError(f"{name} must not contain duplicate pairs")
    return pairs


@dataclass(frozen=True)
class CapabilityFixture:
    schema_version: int
    id: str
    version: int
    dimensions: tuple[str, ...]
    input: CapabilityInput
    subgoals: tuple[CanonicalSubgoal, ...]
    oracle: CapabilityOracle

    def __post_init__(self) -> None:
        if self.schema_version != FIXTURE_SCHEMA_VERSION:
            raise CapabilityFixtureError(f"unsupported fixture schema_version: {self.schema_version!r}")
        _non_empty(self.id, "fixture id")
        _positive_int(self.version, "fixture version")
        dimensions = tuple(self.dimensions)
        if not dimensions or len(dimensions) != len(set(dimensions)) or any(item not in CAPABILITY_DIMENSIONS for item in dimensions):
            raise CapabilityFixtureError("dimensions must be unique known capability dimensions")
        subgoals = tuple(self.subgoals)
        if any(not isinstance(item, CanonicalSubgoal) for item in subgoals):
            raise CapabilityFixtureError("subgoals must contain canonical subgoals")
        if not isinstance(self.input, CapabilityInput) or not isinstance(self.oracle, CapabilityOracle):
            raise CapabilityFixtureError("input and oracle must be capability models")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "subgoals", subgoals)
        self._validate_references()

    def _validate_references(self) -> None:
        action_ids = [item.id for item in self.input.actions]
        subgoal_ids = [item.id for item in self.subgoals]
        if len(action_ids) != len(set(action_ids)):
            raise CapabilityFixtureError("action IDs must be unique")
        if len(subgoal_ids) != len(set(subgoal_ids)):
            raise CapabilityFixtureError("canonical subgoal IDs must be unique")
        actions = set(action_ids)
        subgoals = set(subgoal_ids)
        facts = set(self.input.state)
        if self.input.state_delta is not None and not set(self.input.state_delta) <= facts:
            raise CapabilityFixtureError("state_delta references an unknown state fact")
        for subgoal in self.subgoals:
            if not set(subgoal.action_ids) <= actions:
                raise CapabilityFixtureError("subgoal references unknown action")
        for predicate in (*self.input.constraints, *self.input.policy_forbid,
                          *self.input.policy_require, *self.oracle.replan_rules):
            self._validate_predicate(predicate, actions, subgoals, facts)
        for action in self.input.actions:
            for predicate in action.preconditions:
                self._validate_predicate(predicate, actions, subgoals, facts)
                if predicate.kind not in {"state_equals", "state_exists", "state_absent"}:
                    raise CapabilityFixtureError("action preconditions must be state predicates")
            if any(effect.fact not in facts for effect in action.effects):
                raise CapabilityFixtureError("effect references an unknown state fact")
        required = set(self.oracle.required_subgoal_ids)
        if not required <= subgoals:
            raise CapabilityFixtureError("oracle references unknown required subgoal")
        for edge in (*self.oracle.dependency_edges, *self.oracle.required_order):
            if edge[0] not in subgoals or edge[1] not in subgoals:
                raise CapabilityFixtureError("oracle dependency references unknown subgoal")
        dependency_nodes = {node for edge in self.oracle.dependency_edges for node in edge}
        order_nodes = {node for edge in self.oracle.required_order for node in edge}
        if not dependency_nodes <= required:
            raise CapabilityFixtureError("dependency edges must join required subgoals")
        if not order_nodes <= required:
            raise CapabilityFixtureError("required order must join required subgoals")
        if _has_cycle(required, self.oracle.dependency_edges):
            raise CapabilityFixtureError("dependency graph must be acyclic")
        if _has_cycle(required, (*self.oracle.dependency_edges, *self.oracle.required_order)):
            raise CapabilityFixtureError("dependency and required-order rules must not conflict")
        if self.oracle.action_requirement is ActionRequirement.ACTION_REQUIRED and self.oracle.output_budget.max_steps < len(required):
            raise CapabilityFixtureError("output budget cannot contain the required plan")
        replan_action_rules = [item for item in self.oracle.replan_rules if item.kind in {
            "action_attempted", "action_failed", "action_not_repeated",
        }]
        if replan_action_rules and self.input.prior_result is None:
            raise CapabilityFixtureError("action-history replan rules require prior_result")
        if self.input.prior_result is not None and self.input.prior_result.action_id not in actions:
            raise CapabilityFixtureError("prior result references unknown action")
        if self.input.prior_result is not None:
            for predicate in replan_action_rules:
                if predicate.action_id != self.input.prior_result.action_id:
                    raise CapabilityFixtureError("action-history replan rule must match prior result action")
                if predicate.kind == "action_failed" and self.input.prior_result.outcome not in {"failed", "unavailable"}:
                    raise CapabilityFixtureError("action_failed rule requires failed or unavailable prior result")
        if self.oracle.replan_rules and self.input.prior_result is None and self.input.state_delta is None:
            raise CapabilityFixtureError("replan rules require prior_result or state_delta")
        self._validate_effectful_plan_feasibility()

    @staticmethod
    def _validate_predicate(predicate: Predicate, actions: set[str], subgoals: set[str], facts: set[str]) -> None:
        if predicate.kind in {"state_equals", "state_exists"} and predicate.fact not in facts:
            raise CapabilityFixtureError("predicate references unknown state fact")
        if predicate.kind in {
            "action_selected", "action_not_selected", "action_precondition",
            "action_attempted", "action_failed", "action_not_repeated",
        } and predicate.action_id not in actions:
            raise CapabilityFixtureError("predicate references unknown action")
        if predicate.kind == "subgoal_present" and predicate.subgoal_id not in subgoals:
            raise CapabilityFixtureError("predicate references unknown subgoal")

    def _validate_effectful_plan_feasibility(self) -> None:
        required = self.oracle.required_subgoal_ids
        if not required:
            return
        by_id = {item.id: item for item in self.subgoals}
        selected = [by_id[item] for item in required]
        if not any(action.effects for subgoal in selected for action in self.input.actions if action.id in subgoal.action_ids):
            return
        if any(len(subgoal.action_ids) != 1 for subgoal in selected):
            return  # Alternative action choice belongs to evaluator work, not authoring search.
        action_by_id = {item.id: item for item in self.input.actions}
        edges = self.oracle.dependency_edges
        for order in _topological_orders(required, edges):
            state = thaw_json(self.input.state)
            if self.input.state_delta is not None:
                state.update(thaw_json(self.input.state_delta))
            for subgoal_id in order:
                action = action_by_id[by_id[subgoal_id].action_ids[0]]
                if not all(_state_predicate_holds(item, state) for item in action.preconditions):
                    raise CapabilityFixtureError("effectful required plan is not feasible under declared transitions")
                for effect in action.effects:
                    state[effect.fact] = thaw_json(effect.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "version": self.version,
            "dimensions": list(self.dimensions),
            "input": self.input.to_dict(),
            "subgoals": [item.to_dict() for item in self.subgoals],
            "oracle": self.oracle.to_dict(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_fixture_bytes(self)

    @property
    def digest(self) -> str:
        return fixture_digest(self)


def _state_predicate_holds(predicate: Predicate, state: Mapping[str, Any]) -> bool:
    if predicate.kind == "state_equals":
        return state[predicate.fact] == thaw_json(predicate.value)
    if predicate.kind == "state_exists":
        return predicate.fact in state
    if predicate.kind == "state_absent":
        return predicate.fact not in state
    raise CapabilityFixtureError("non-state predicate cannot be an action precondition")


def _has_cycle(nodes: set[str], edges: Sequence[tuple[str, str]]) -> bool:
    children = {node: set() for node in nodes}
    for source, target in edges:
        children[source].add(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        found = any(visit(child) for child in children[node])
        visiting.remove(node)
        visited.add(node)
        return found

    return any(visit(node) for node in nodes)


def _topological_orders(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    """Enumerate the tiny fixture DAG only; this is validation, not planning."""
    node_set = set(nodes)
    predecessors = {node: set() for node in nodes}
    for source, target in edges:
        predecessors[target].add(source)
    orders: list[tuple[str, ...]] = []

    def build(prefix: tuple[str, ...], remaining: set[str]) -> None:
        if not remaining:
            orders.append(prefix)
            return
        ready = sorted(node for node in remaining if predecessors[node] <= set(prefix))
        for node in ready:
            build((*prefix, node), remaining - {node})

    build((), node_set)
    return tuple(orders)


def canonical_fixture_bytes(fixture: CapabilityFixture) -> bytes:
    if not isinstance(fixture, CapabilityFixture):
        raise CapabilityFixtureError("fixture must be CapabilityFixture")
    return json.dumps(
        fixture.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fixture_digest(fixture: CapabilityFixture) -> str:
    return hashlib.sha256(canonical_fixture_bytes(fixture)).hexdigest()
