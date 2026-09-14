"""Pure deterministic Capability Round 2 response evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Mapping

from .models import CapabilityFixture, CapabilityFixtureError, Predicate, thaw_json

EVALUATOR_VERSION = 1
FAILURE_CODE_VERSION = 1

class EvaluationState(str, Enum): PASS="PASS"; FAIL="FAIL"; NOT_EVALUABLE="NOT_EVALUABLE"; NOT_APPLICABLE="NOT_APPLICABLE"
class CapabilityVerdict(str, Enum): CAPABILITY_VALID="CAPABILITY_VALID"; CAPABILITY_INVALID="CAPABILITY_INVALID"; NOT_EVALUATED="NOT_EVALUATED"
class EvaluationStatus(str, Enum): EVALUATED="EVALUATED"; FIXTURE_INVALID="FIXTURE_INVALID"
class FailureCode(str, Enum):
    INVALID_JSON="INVALID_JSON"; SCHEMA_INVALID="SCHEMA_INVALID"; CONSTRAINT_VIOLATION="CONSTRAINT_VIOLATION"; POLICY_PROHIBITED_ACTION="POLICY_PROHIBITED_ACTION"; POLICY_BOUNDARY_VIOLATION="POLICY_BOUNDARY_VIOLATION"; UNKNOWN_ACTION="UNKNOWN_ACTION"; UNMET_PRECONDITION="UNMET_PRECONDITION"; ACTION_WHEN_NO_ACTION_REQUIRED="ACTION_WHEN_NO_ACTION_REQUIRED"; NO_ACTION_WHEN_ACTION_REQUIRED="NO_ACTION_WHEN_ACTION_REQUIRED"; UNKNOWN_SUBGOAL="UNKNOWN_SUBGOAL"; MISSING_REQUIRED_SUBGOAL="MISSING_REQUIRED_SUBGOAL"; DUPLICATE_SUBGOAL="DUPLICATE_SUBGOAL"; INVALID_DEPENDENCY="INVALID_DEPENDENCY"; DEPENDENCY_CYCLE="DEPENDENCY_CYCLE"; REPEATED_FAILED_ACTION="REPEATED_FAILED_ACTION"; IGNORED_STATE_CHANGE="IGNORED_STATE_CHANGE"; OUTPUT_BUDGET_EXCEEDED="OUTPUT_BUDGET_EXCEEDED"

@dataclass(frozen=True)
class CapabilityEvaluationResult:
    fixture_id: str; fixture_version: int; fixture_digest: str; evaluator_version: int
    failure_code_version: int; status: EvaluationStatus; verdict: CapabilityVerdict
    response_bytes: int; parse_state: EvaluationState; schema_state: EvaluationState
    hard_gate_results: Mapping[str, EvaluationState]; failure_codes: tuple[FailureCode, ...]
    graded_measurements: Mapping[str, Any]; parsed_response: Mapping[str, Any] | None
    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_gate_results", MappingProxyType(dict(self.hard_gate_results)))
        object.__setattr__(self, "graded_measurements", MappingProxyType(dict(self.graded_measurements)))
        object.__setattr__(self, "parsed_response", None if self.parsed_response is None else MappingProxyType(dict(self.parsed_response)))

def _state(value: bool) -> EvaluationState: return EvaluationState.PASS if value else EvaluationState.FAIL
def _cycle(edges: list[tuple[str,str]]) -> bool:
    graph: dict[str,set[str]]={}
    for a,b in edges: graph.setdefault(a,set()).add(b); graph.setdefault(b,set())
    seen:set[str]=set(); active:set[str]=set()
    def visit(n:str)->bool:
        if n in active:return True
        if n in seen:return False
        active.add(n); found=any(visit(x) for x in graph[n]); active.remove(n);seen.add(n);return found
    return any(visit(n) for n in graph)

def _predicate(predicate: Predicate, *, state: Mapping[str,Any], actions:set[str], subgoals:set[str], mode:str, fixture:CapabilityFixture)->bool:
    if predicate.kind=="state_equals": return state.get(predicate.fact, object()) == thaw_json(predicate.value)
    if predicate.kind=="state_exists": return predicate.fact in state
    if predicate.kind=="state_absent": return predicate.fact not in state
    if predicate.kind=="action_selected": return predicate.action_id in actions
    if predicate.kind=="action_not_selected": return predicate.action_id not in actions
    if predicate.kind=="action_precondition":
        action=next(item for item in fixture.input.actions if item.id==predicate.action_id)
        return all(_predicate(x,state=state,actions=actions,subgoals=subgoals,mode=mode,fixture=fixture) for x in action.preconditions)
    if predicate.kind=="action_attempted": return fixture.input.prior_result is not None and fixture.input.prior_result.action_id==predicate.action_id
    if predicate.kind=="action_failed": return fixture.input.prior_result is not None and fixture.input.prior_result.action_id==predicate.action_id and fixture.input.prior_result.outcome in {"failed","unavailable"}
    if predicate.kind=="action_not_repeated": return predicate.action_id not in actions
    if predicate.kind=="subgoal_present": return predicate.subgoal_id in subgoals
    return mode=="NO_ACTION"

def evaluate(fixture: CapabilityFixture, raw_response: str) -> CapabilityEvaluationResult:
    """Evaluate exact response text without repair, inference, or external state."""
    try:
        if not isinstance(fixture, CapabilityFixture): raise CapabilityFixtureError("fixture type")
        # Construction already performs author validation; force complete semantic serialization too.
        digest=fixture.digest
    except Exception:
        return CapabilityEvaluationResult("",0,"",EVALUATOR_VERSION,FAILURE_CODE_VERSION,EvaluationStatus.FIXTURE_INVALID,CapabilityVerdict.NOT_EVALUATED,0,EvaluationState.NOT_EVALUABLE,EvaluationState.NOT_EVALUABLE,{},(),{},None)
    if not isinstance(raw_response,str): raw_response=""
    response_bytes=len(raw_response.encode("utf-8"))
    try: parsed=json.loads(raw_response)
    except (json.JSONDecodeError,UnicodeEncodeError):
        return _protocol(fixture,digest,response_bytes,FailureCode.INVALID_JSON,EvaluationState.FAIL)
    if not isinstance(parsed,dict) or set(parsed)!={"mode","plan"} or parsed.get("mode") not in {"ACT","NO_ACTION"} or not isinstance(parsed.get("plan"),list):
        return _protocol(fixture,digest,response_bytes,FailureCode.SCHEMA_INVALID,EvaluationState.PASS)
    plan=parsed["plan"]
    if any(not isinstance(step,dict) or set(step)!={"subgoal","action","depends_on"} or not isinstance(step["subgoal"],str) or not isinstance(step["action"],str) or not isinstance(step["depends_on"],list) or any(not isinstance(x,str) for x in step["depends_on"]) for step in plan):
        return _protocol(fixture,digest,response_bytes,FailureCode.SCHEMA_INVALID,EvaluationState.PASS)
    mode=parsed["mode"]; codes:list[FailureCode]=[]; gates:dict[str,EvaluationState]={"structured_protocol":EvaluationState.PASS}
    if fixture.oracle.action_requirement.value=="NO_ACTION" and mode!="NO_ACTION": codes.append(FailureCode.ACTION_WHEN_NO_ACTION_REQUIRED)
    if fixture.oracle.action_requirement.value=="ACTION_REQUIRED" and mode!="ACT": codes.append(FailureCode.NO_ACTION_WHEN_ACTION_REQUIRED)
    if (mode=="NO_ACTION" and plan) or (mode=="ACT" and not plan): codes.append(FailureCode.SCHEMA_INVALID); gates["structured_protocol"]=EvaluationState.FAIL
    subgoal_defs={x.id:x for x in fixture.subgoals}; action_defs={x.id:x for x in fixture.input.actions}
    known_steps=[]; actions=set(); subgoals=set(); failed_steps:set[str]=set(); state=thaw_json(fixture.input.state)
    if fixture.input.state_delta: state.update(thaw_json(fixture.input.state_delta))
    for step in plan:
        sg,act=step["subgoal"],step["action"]
        if sg not in subgoal_defs: codes.append(FailureCode.UNKNOWN_SUBGOAL); continue
        if act not in action_defs: codes.append(FailureCode.UNKNOWN_ACTION); continue
        subgoals.add(sg); actions.add(act); known_steps.append(step)
        if act not in subgoal_defs[sg].action_ids: codes.append(FailureCode.INVALID_DEPENDENCY)
        upstream=set(step["depends_on"])
        if any(x not in subgoal_defs for x in upstream): codes.append(FailureCode.INVALID_DEPENDENCY)
        if upstream & failed_steps: continue
        action=action_defs[act]
        if not all(_predicate(p,state=state,actions=actions,subgoals=subgoals,mode=mode,fixture=fixture) for p in action.preconditions):
            codes.append(FailureCode.UNMET_PRECONDITION); failed_steps.add(sg)
            if fixture.input.state_delta: codes.append(FailureCode.IGNORED_STATE_CHANGE)
            continue
        for effect in action.effects: state[effect.fact]=thaw_json(effect.value)
    required=set(fixture.oracle.required_subgoal_ids)
    counts={x:sum(step["subgoal"]==x for step in plan) for x in required}
    if any(v==0 for v in counts.values()): codes.append(FailureCode.MISSING_REQUIRED_SUBGOAL)
    duplicates=sum(max(0,v-1) for v in counts.values())
    if duplicates: codes.append(FailureCode.DUPLICATE_SUBGOAL)
    response_edges=[(dep,step["subgoal"]) for step in plan for dep in step["depends_on"] if dep in subgoal_defs]
    if _cycle(response_edges): codes.append(FailureCode.DEPENDENCY_CYCLE)
    expected=set(fixture.oracle.dependency_edges); got=set(response_edges)
    if not expected <= got or any(edge not in expected for edge in got): codes.append(FailureCode.INVALID_DEPENDENCY)
    positions={step["subgoal"]:i for i,step in enumerate(plan) if step["subgoal"] not in {x for x in []}}
    order_ok=sum(1 for a,b in fixture.oracle.required_order if positions.get(a,-1)<positions.get(b,-1))
    constraint_ok=all(_predicate(p,state=state,actions=actions,subgoals=subgoals,mode=mode,fixture=fixture) for p in fixture.input.constraints)
    forbid_ok=all(_predicate(p,state=state,actions=actions,subgoals=subgoals,mode=mode,fixture=fixture) for p in fixture.input.policy_forbid)
    require_ok=all(_predicate(p,state=state,actions=actions,subgoals=subgoals,mode=mode,fixture=fixture) for p in fixture.input.policy_require)
    if not constraint_ok: codes.append(FailureCode.CONSTRAINT_VIOLATION)
    if not forbid_ok: codes.append(FailureCode.POLICY_PROHIBITED_ACTION)
    if not require_ok: codes.append(FailureCode.POLICY_BOUNDARY_VIOLATION)
    if any(c in codes for c in {FailureCode.UNKNOWN_ACTION,FailureCode.UNMET_PRECONDITION,FailureCode.ACTION_WHEN_NO_ACTION_REQUIRED,FailureCode.NO_ACTION_WHEN_ACTION_REQUIRED}): gates["state_grounded_action"]=EvaluationState.FAIL
    else:gates["state_grounded_action"]=EvaluationState.PASS
    gates["constraint_policy"]= _state(not any(c in codes for c in {FailureCode.CONSTRAINT_VIOLATION,FailureCode.POLICY_PROHIBITED_ACTION,FailureCode.POLICY_BOUNDARY_VIOLATION}))
    if FailureCode.SCHEMA_INVALID in codes:gates["structured_protocol"]=EvaluationState.FAIL
    if response_bytes>fixture.oracle.output_budget.max_bytes or len(plan)>fixture.oracle.output_budget.max_steps: codes.append(FailureCode.OUTPUT_BUDGET_EXCEEDED)
    for rule in fixture.oracle.replan_rules:
        if not _predicate(rule,state=state,actions=actions,subgoals=subgoals,mode=mode,fixture=fixture):
            codes.append(FailureCode.REPEATED_FAILED_ACTION if rule.kind=="action_not_repeated" else FailureCode.IGNORED_STATE_CHANGE)
    codes=list(dict.fromkeys(codes))
    verdict=CapabilityVerdict.CAPABILITY_INVALID if any(v is EvaluationState.FAIL for v in gates.values()) else CapabilityVerdict.CAPABILITY_VALID
    measures={"required_subgoals_present":sum(v==1 for v in counts.values()),"required_subgoals_total":len(required),"duplicate_count":duplicates,"dependency_edges_satisfied":len(expected&got),"dependency_edges_total":len(expected),"dag_valid":not _cycle(response_edges),"required_order_pairs_satisfied":order_ok,"required_order_pairs_total":len(fixture.oracle.required_order),"repeated_barred_action_count":sum(c is FailureCode.REPEATED_FAILED_ACTION for c in codes),"response_bytes":response_bytes,"plan_step_count":len(plan),"byte_limit":fixture.oracle.output_budget.max_bytes,"step_limit":fixture.oracle.output_budget.max_steps,"output_budget_compliant":FailureCode.OUTPUT_BUDGET_EXCEEDED not in codes}
    return CapabilityEvaluationResult(fixture.id,fixture.version,digest,EVALUATOR_VERSION,FAILURE_CODE_VERSION,EvaluationStatus.EVALUATED,verdict,response_bytes,EvaluationState.PASS,EvaluationState.PASS,gates,tuple(codes),measures,parsed)

def _protocol(fixture:CapabilityFixture,digest:str,size:int,code:FailureCode,parse:EvaluationState)->CapabilityEvaluationResult:
    codes=(code,) if size<=fixture.oracle.output_budget.max_bytes else (code,FailureCode.OUTPUT_BUDGET_EXCEEDED)
    measures={"response_bytes":size,"plan_step_count":None,"byte_limit":fixture.oracle.output_budget.max_bytes,"step_limit":fixture.oracle.output_budget.max_steps,"output_budget_compliant":len(codes)==1}
    return CapabilityEvaluationResult(fixture.id,fixture.version,digest,EVALUATOR_VERSION,FAILURE_CODE_VERSION,EvaluationStatus.EVALUATED,CapabilityVerdict.CAPABILITY_INVALID,size,parse,EvaluationState.NOT_EVALUABLE if code is FailureCode.INVALID_JSON else EvaluationState.FAIL,{"structured_protocol":EvaluationState.FAIL,"constraint_policy":EvaluationState.NOT_EVALUABLE,"state_grounded_action":EvaluationState.NOT_EVALUABLE},codes,measures,None)
