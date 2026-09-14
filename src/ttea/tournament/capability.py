"""Fixture-scoped immutable Capability Round 2 work persistence."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib, json, os
from pathlib import Path
from uuid import uuid4
from typing import Any
from ttea.capability.evaluator import CapabilityEvaluationResult
from ttea.capability.evaluator import CapabilityVerdict, EvaluationState
from ttea.capability.catalogue import CAPABILITY_FIXTURES

ROUND2_CANDIDATES=("qwen35-08b","granite4-3b-h","granite33-2b","deepseek-r1-15b")
PLANNING_FIXTURES=frozenset(("cap.plan.chain","cap.plan.fork_join"))
REPLANNING_FIXTURES=frozenset(("cap.replan.failed_alternative","cap.replan.state_delta"))
class CapabilityTerminalState(str):
    COMPLETED_ELIGIBLE="COMPLETED_ELIGIBLE"; COMPLETED_NO_ELIGIBLE_MODEL="COMPLETED_NO_ELIGIBLE_MODEL"; COMPLETED_INCOMPLETE="COMPLETED_INCOMPLETE"; BLOCKED_INTEGRITY="BLOCKED_INTEGRITY"
@dataclass(frozen=True)
class CandidateCapabilitySummary:
    candidate_id:str; status:str; planning:float|None; replanning:float|None; final_score:float|None; response_bytes:int; plan_steps:int; failures:tuple[str,...]
def capability_work_order(tournament_id:str):
    return tuple(CapabilityWorkId(tournament_id,c,f.id,f.version) for c in ROUND2_CANDIDATES for f in CAPABILITY_FIXTURES)
def _ratio(n:int,d:int)->float|None:return None if d==0 else n/d
def summarize_candidate(candidate_id:str, results:dict[str,CapabilityEvaluationResult], *, incomplete:bool=False)->CandidateCapabilitySummary:
    if incomplete or set(results)!={f.id for f in CAPABILITY_FIXTURES}:return CandidateCapabilitySummary(candidate_id,"INCOMPLETE",None,None,None,0,0,())
    codes=tuple(code.value for r in results.values() for code in r.failure_codes)
    if any(r.verdict is not CapabilityVerdict.CAPABILITY_VALID for r in results.values()):return CandidateCapabilitySummary(candidate_id,"CAPABILITY_INVALID",None,None,None,sum(r.response_bytes for r in results.values()),sum((r.graded_measurements.get("plan_step_count") or 0) for r in results.values()),codes)
    def totals(ids):
        rs=[results[x] for x in ids];m=[r.graded_measurements for r in rs]
        cov=_ratio(sum(x["required_subgoals_present"] for x in m),sum(x["required_subgoals_total"] for x in m));edge=_ratio(sum(x["dependency_edges_satisfied"] for x in m),sum(x["dependency_edges_total"] for x in m));dag=sum(bool(x["dag_valid"]) for x in m)/len(m);order=_ratio(sum(x["required_order_pairs_satisfied"] for x in m),sum(x["required_order_pairs_total"] for x in m));return cov,edge,dag,order,m
    c,e,d,o,_=totals(PLANNING_FIXTURES)
    if None in (c,e,o):return CandidateCapabilitySummary(candidate_id,"INCOMPLETE",None,None,None,0,0,codes)
    planning=25*(c+e+d+o)
    rs=[results[x].graded_measurements for x in REPLANNING_FIXTURES]
    # Frozen fixtures expose adaptation through barred-action cleanliness and changed-state compliance.
    adapt=_ratio(sum(x.get("adaptation_invariants_satisfied",1) for x in rs),sum(x.get("adaptation_invariants_total",1) for x in rs));changed=_ratio(sum(x.get("state_change_requirements_satisfied",1) for x in rs),sum(x.get("state_change_requirements_total",1) for x in rs));clean=sum(x.get("repeated_barred_action_count",0)==0 for x in rs)/len(rs)
    replanning=40*adapt+40*changed+20*clean
    b=sum(r.response_bytes for r in results.values());s=sum((r.graded_measurements.get("plan_step_count") or 0) for r in results.values())
    return CandidateCapabilitySummary(candidate_id,"ELIGIBLE",planning,replanning,.5*(planning+replanning),b,s,codes)
def rank_candidates(items):return tuple(sorted((x for x in items if x.status=="ELIGIBLE"),key=lambda x:(-x.final_score,-x.replanning,-x.planning,x.response_bytes,x.plan_steps,x.candidate_id)))
def terminal_state(items,integrity_blocker:bool=False):
    if integrity_blocker:return CapabilityTerminalState.BLOCKED_INTEGRITY
    if any(x.status=="ELIGIBLE" for x in items):return CapabilityTerminalState.COMPLETED_ELIGIBLE
    if any(x.status=="INCOMPLETE" for x in items):return CapabilityTerminalState.COMPLETED_INCOMPLETE
    return CapabilityTerminalState.COMPLETED_NO_ELIGIBLE_MODEL

class CapabilityStoreError(RuntimeError): pass
def _text(x:object,n:str)->None:
    if not isinstance(x,str) or not x.strip(): raise CapabilityStoreError(f"{n} must be non-empty")
@dataclass(frozen=True)
class CapabilityWorkId:
    tournament_id:str; candidate_id:str; fixture_id:str; fixture_version:int
    def __post_init__(self):
        for n in ("tournament_id","candidate_id","fixture_id"): _text(getattr(self,n),n)
        if not isinstance(self.fixture_version,int) or isinstance(self.fixture_version,bool) or self.fixture_version<=0: raise CapabilityStoreError("fixture_version must be positive")
    @property
    def key(self)->str:return hashlib.sha256((self.tournament_id+"\0"+self.candidate_id+"\0"+self.fixture_id+"\0"+str(self.fixture_version)).encode()).hexdigest()
    def to_dict(self):return {"tournament_id":self.tournament_id,"candidate_id":self.candidate_id,"fixture_id":self.fixture_id,"fixture_version":self.fixture_version}
@dataclass(frozen=True)
class RawCapabilityResponse:
    work_id:CapabilityWorkId; fixture_digest:str; technical_attempt_id:str; response_utf8:str; response_sha256:str; response_bytes:int
    def __post_init__(self):
        _text(self.fixture_digest,"fixture_digest");_text(self.technical_attempt_id,"technical_attempt_id")
        if not isinstance(self.response_utf8,str):raise CapabilityStoreError("response must be text")
        raw=self.response_utf8.encode("utf-8"); digest=hashlib.sha256(raw).hexdigest()
        if self.response_sha256!=digest or self.response_bytes!=len(raw):raise CapabilityStoreError("raw response digest or byte count mismatch")
    @classmethod
    def create(cls,work_id,fixture_digest,attempt_id,response):
        raw=response.encode("utf-8");return cls(work_id,fixture_digest,attempt_id,response,hashlib.sha256(raw).hexdigest(),len(raw))
@dataclass(frozen=True)
class CapabilityWorkRecord:
    work_id:CapabilityWorkId; fixture_digest:str; technical_attempt_count:int=0; active_attempt_id:str|None=None; raw_response_digest:str|None=None; technical_evidence_refs:tuple[str,...]=(); evaluator_result_digest:str|None=None; terminal_status:str|None=None; terminal_reason:str|None=None
    @property
    def phase(self)->str:
        if self.terminal_status:return "TERMINAL"
        if self.evaluator_result_digest:return "EVALUATOR_RESULT_PERSISTED"
        if self.technical_evidence_refs:return "CLEANUP_COMPLETE"
        if self.raw_response_digest:return "RAW_RESPONSE_PERSISTED"
        return "TECHNICAL_EXECUTION" if self.active_attempt_id else "PENDING"

def _raw_to_dict(r):return {"work_id":r.work_id.to_dict(),"fixture_digest":r.fixture_digest,"technical_attempt_id":r.technical_attempt_id,"response_utf8":r.response_utf8,"response_sha256":r.response_sha256,"response_bytes":r.response_bytes}
def _result_to_dict(r:CapabilityEvaluationResult):return {"fixture_id":r.fixture_id,"fixture_version":r.fixture_version,"fixture_digest":r.fixture_digest,"evaluator_version":r.evaluator_version,"failure_code_version":r.failure_code_version,"status":r.status.value,"verdict":r.verdict.value,"response_bytes":r.response_bytes,"parse_state":r.parse_state.value,"schema_state":r.schema_state.value,"hard_gate_results":{k:v.value for k,v in r.hard_gate_results.items()},"failure_codes":[x.value for x in r.failure_codes],"graded_measurements":dict(r.graded_measurements),"parsed_response":None if r.parsed_response is None else dict(r.parsed_response)}

class CapabilityWorkStore:
    def __init__(self,root:Path|str):self.root=Path(root)
    def _dir(self,w:CapabilityWorkId)->Path:return self.root/w.tournament_id/"capability"/w.key
    def _path(self,w,n):return self._dir(w)/n
    def _atomic(self,path:Path,payload:dict,*,immutable:bool=False)->None:
        path.parent.mkdir(parents=True,exist_ok=True)
        encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
        if immutable and path.exists():
            if path.read_bytes()!=encoded:raise CapabilityStoreError("immutable capability evidence conflict")
            return
        tmp=path.with_name("."+path.name+"."+uuid4().hex+".tmp")
        try:
            with tmp.open("xb") as h:h.write(encoded);h.flush();os.fsync(h.fileno())
            if immutable:
                try:os.link(tmp,path)
                except FileExistsError:
                    if path.read_bytes()!=encoded:raise CapabilityStoreError("immutable capability evidence conflict")
            else:os.replace(tmp,path)
        finally:tmp.unlink(missing_ok=True)
    def create(self,w:CapabilityWorkId,fixture_digest:str)->CapabilityWorkRecord:
        if self._path(w,"work.json").is_file():
            existing=self.load(w)
            if existing.fixture_digest!=fixture_digest:raise CapabilityStoreError("capability work identity conflict")
            return existing
        r=CapabilityWorkRecord(w,fixture_digest);self._atomic(self._path(w,"work.json"),self._work(r));return r
    def _work(self,r):return {"work_id":r.work_id.to_dict(),"fixture_digest":r.fixture_digest,"technical_attempt_count":r.technical_attempt_count,"active_attempt_id":r.active_attempt_id,"raw_response_digest":r.raw_response_digest,"technical_evidence_refs":list(r.technical_evidence_refs),"evaluator_result_digest":r.evaluator_result_digest,"terminal_status":r.terminal_status,"terminal_reason":r.terminal_reason}
    def load(self,w):
        p=self._path(w,"work.json")
        if not p.is_file():raise CapabilityStoreError("unknown capability work")
        x=json.loads(p.read_text(encoding="utf-8")); ident=CapabilityWorkId(**x["work_id"])
        if ident!=w:raise CapabilityStoreError("work path identity mismatch")
        return CapabilityWorkRecord(ident,x["fixture_digest"],x["technical_attempt_count"],x.get("active_attempt_id"),x.get("raw_response_digest"),tuple(x.get("technical_evidence_refs",())),x.get("evaluator_result_digest"),x.get("terminal_status"),x.get("terminal_reason"))
    def save(self,r):self._atomic(self._path(r.work_id,"work.json"),self._work(r));return r
    def begin_attempt(self,w):
        r=self.load(w)
        if r.raw_response_digest:raise CapabilityStoreError("raw response exists; generation prohibited")
        if r.active_attempt_id:raise CapabilityStoreError("technical attempt already active")
        n=r.technical_attempt_count+1; return self.save(replace(r,technical_attempt_count=n,active_attempt_id=f"{w.key}:attempt:{n}"))
    def publish_raw(self,raw:RawCapabilityResponse):
        r=self.load(raw.work_id)
        if r.raw_response_digest is not None:
            existing=self.raw_response(raw.work_id)
            if existing!=raw:raise CapabilityStoreError("immutable capability evidence conflict")
            return r
        if r.fixture_digest!=raw.fixture_digest or r.active_attempt_id!=raw.technical_attempt_id:raise CapabilityStoreError("raw response work identity mismatch")
        self._atomic(self._path(raw.work_id,"raw-response.json"),_raw_to_dict(raw),immutable=True)
        return self.save(replace(r,raw_response_digest=raw.response_sha256))
    def raw_response(self,w):
        x=json.loads(self._path(w,"raw-response.json").read_text(encoding="utf-8"));raw=RawCapabilityResponse(CapabilityWorkId(**x["work_id"]),x["fixture_digest"],x["technical_attempt_id"],x["response_utf8"],x["response_sha256"],x["response_bytes"])
        if raw.work_id!=w:raise CapabilityStoreError("raw response path identity mismatch")
        return raw
    def mark_cleanup_complete(self,w,evidence_ref:str):
        r=self.load(w);_text(evidence_ref,"evidence_ref")
        return self.save(replace(r,active_attempt_id=None,technical_evidence_refs=tuple((*r.technical_evidence_refs,evidence_ref))))
    def publish_result(self,w,result:CapabilityEvaluationResult):
        r=self.load(w)
        if not r.raw_response_digest or result.fixture_digest!=r.fixture_digest or (result.fixture_id,result.fixture_version)!=(w.fixture_id,w.fixture_version):raise CapabilityStoreError("evaluator result binding mismatch")
        payload={"work_id":w.to_dict(),"raw_response_digest":r.raw_response_digest,"result":_result_to_dict(result)};digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        self._atomic(self._path(w,"evaluator-result.json"),payload,immutable=True)
        return self.save(replace(r,evaluator_result_digest=digest))
    def recover(self,w):
        r=self.load(w)
        if r.active_attempt_id:return self.save(replace(r,active_attempt_id=None))
        return r
