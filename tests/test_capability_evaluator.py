from __future__ import annotations
import json, tempfile, unittest
from dataclasses import replace
from pathlib import Path
from ttea.capability import CAPABILITY_FIXTURES, CapabilityVerdict, FailureCode, evaluate
from ttea.tournament.capability import CapabilityStoreError, CapabilityWorkId, CapabilityWorkStore, RawCapabilityResponse

def fixture(name): return next(x for x in CAPABILITY_FIXTURES if x.id==name)
def response(mode, plan): return json.dumps({"mode":mode,"plan":plan},separators=(",",":"))
def step(s,a,deps=()): return {"subgoal":s,"action":a,"depends_on":list(deps)}

class EvaluatorTests(unittest.TestCase):
 def test_valid_act_no_action_and_replay(self):
  a=evaluate(fixture("cap.protocol.basic"),response("ACT",[step("prepare","prepare_output")]))
  n=evaluate(fixture("cap.state.no_action"),response("NO_ACTION",[]))
  self.assertEqual(a.verdict,CapabilityVerdict.CAPABILITY_VALID);self.assertEqual(n.verdict,CapabilityVerdict.CAPABILITY_VALID);self.assertEqual(a,evaluate(fixture("cap.protocol.basic"),response("ACT",[step("prepare","prepare_output")])))
 def test_protocol_constraint_policy_and_precondition_failures(self):
  self.assertIn(FailureCode.INVALID_JSON,evaluate(fixture("cap.protocol.basic"),"no").failure_codes)
  self.assertIn(FailureCode.SCHEMA_INVALID,evaluate(fixture("cap.protocol.basic"),"{}").failure_codes)
  self.assertIn(FailureCode.CONSTRAINT_VIOLATION,evaluate(fixture("cap.constraint.preserve_input"),response("ACT",[step("prepare","discard_input")])).failure_codes)
  self.assertIn(FailureCode.POLICY_PROHIBITED_ACTION,evaluate(fixture("cap.policy.guard_boundary"),response("ACT",[step("obtain_approval","override_guard")])).failure_codes)
  self.assertIn(FailureCode.UNMET_PRECONDITION,evaluate(fixture("cap.state.precondition_choice"),response("ACT",[step("commit","fast_commit")])).failure_codes)
 def test_chain_and_fork_orders(self):
  c=response("ACT",[step("collect","inspect_input"),step("validate","validate_item",["collect"]),step("prepare","prepare_output",["validate"]),step("commit","commit_output",["prepare"])])
  self.assertEqual(evaluate(fixture("cap.plan.chain"),c).verdict,CapabilityVerdict.CAPABILITY_VALID)
  for first,second in (("collect_input","collect_policy"),("collect_policy","collect_input")):
   actions={"collect_input":"inspect_input","collect_policy":"inspect_policy"}
   p=response("ACT",[step(first,actions[first]),step(second,actions[second]),step("validate","validate_item",["collect_input","collect_policy"]),step("prepare","prepare_output",["validate"])])
   self.assertEqual(evaluate(fixture("cap.plan.fork_join"),p).verdict,CapabilityVerdict.CAPABILITY_VALID)
 def test_plan_replan_and_budget_findings(self):
  invented=response("ACT",[step("collect_input","inspect_input"),step("collect_policy","inspect_policy",["collect_input"]),step("validate","validate_item",["collect_input","collect_policy"]),step("prepare","prepare_output",["validate"])])
  self.assertIn(FailureCode.INVALID_DEPENDENCY,evaluate(fixture("cap.plan.fork_join"),invented).failure_codes)
  repeated=response("ACT",[step("prepare","prepare_primary")]);self.assertIn(FailureCode.REPEATED_FAILED_ACTION,evaluate(fixture("cap.replan.failed_alternative"),repeated).failure_codes)
  obsolete=response("ACT",[step("commit","prepare_output")]);self.assertIn(FailureCode.IGNORED_STATE_CHANGE,evaluate(fixture("cap.replan.state_delta"),obsolete).failure_codes)
  self.assertIn(FailureCode.OUTPUT_BUDGET_EXCEEDED,evaluate(fixture("cap.protocol.basic"),"x"*513).failure_codes)
 def test_missing_duplicates_cycles_and_independent_findings(self):
  missing=response("ACT",[step("collect","inspect_input")])
  self.assertIn(FailureCode.MISSING_REQUIRED_SUBGOAL,evaluate(fixture("cap.plan.chain"),missing).failure_codes)
  duplicate=response("ACT",[step("prepare","prepare_output"),step("prepare","prepare_output")])
  self.assertIn(FailureCode.DUPLICATE_SUBGOAL,evaluate(fixture("cap.protocol.basic"),duplicate).failure_codes)
  cycle=response("ACT",[step("collect","inspect_input",["validate"]),step("validate","validate_item",["collect"]),step("prepare","prepare_output",["validate"]),step("commit","commit_output",["prepare"])])
  self.assertIn(FailureCode.DEPENDENCY_CYCLE,evaluate(fixture("cap.plan.chain"),cycle).failure_codes)

class CapabilityStoreTests(unittest.TestCase):
 def test_raw_result_recovery_conflict_and_isolation(self):
  with tempfile.TemporaryDirectory() as d:
   store=CapabilityWorkStore(Path(d)); f=fixture("cap.protocol.basic"); w=CapabilityWorkId("t","a",f.id,f.version);store.create(w,f.digest)
   active=store.begin_attempt(w);raw=RawCapabilityResponse.create(w,f.digest,active.active_attempt_id,response("ACT",[step("prepare","prepare_output")]))
   saved=store.publish_raw(raw);self.assertEqual(saved.raw_response_digest,raw.response_sha256);self.assertEqual(store.publish_raw(raw).raw_response_digest,raw.response_sha256)
   self.assertEqual(store.recover(w).phase,"RAW_RESPONSE_PERSISTED")
   with self.assertRaises(CapabilityStoreError):store.begin_attempt(w)
   store.mark_cleanup_complete(w,"tech-1");result=evaluate(f,raw.response_utf8);self.assertTrue(store.publish_result(w,result).evaluator_result_digest)
   self.assertTrue(store.publish_result(w,result).evaluator_result_digest)
   with self.assertRaises(CapabilityStoreError):store.publish_result(w,replace(result,response_bytes=1))
   with self.assertRaises(CapabilityStoreError):store.publish_raw(RawCapabilityResponse.create(w,f.digest,active.active_attempt_id,"{}"))
   other=CapabilityWorkId("t","b",f.id,f.version);store.create(other,f.digest);self.assertEqual(store.load(other).phase,"PENDING")

if __name__=="__main__":unittest.main()
