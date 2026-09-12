"""Persistent detached supervision for autonomous validation campaigns."""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable
from uuid import uuid4

import psutil

from ttea.benchmarking import BenchmarkStateStore
from ttea.benchmarking.ollama import OwnedProcessIdentity
from ttea.resources import ResourceDecision, ResourceManager, ResourceRequest
from ttea.telemetry import HardwareTelemetryService

from .campaign_models import (ACTIVE_CAMPAIGN_STATES, CampaignRunRecord, CampaignState, CampaignWaitPeriod,
                              ValidationCampaign, campaign_from_dict, campaign_to_dict)
from .models import ValidationRunResult, ValidationStatus
from .reporting import result_to_dict
from .runner import SuiteRegistry, ValidationRunner
from .suites import MIB, build_registry, stop_verified_process


CAMPAIGN_SUITE_ORDER = ("pre_model", "tiny_model", "model_tournament", "runtime_regression", "full_system")
DEFAULT_CAMPAIGN_ROOT = Path("data/validation/campaigns")
DEFAULT_REPORT_ROOT = Path("data/validation")
DEFAULT_STATE_PATH = Path("data/runtime/benchmark.json")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CampaignStore:
    """Atomic campaign persistence plus one-active-campaign arbitration."""

    def __init__(self, root: Path = DEFAULT_CAMPAIGN_ROOT,
                 process_matches: Callable[[object, object], bool] = BenchmarkStateStore.process_matches) -> None:
        self.root, self.process_matches = root, process_matches

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def campaign_dir(self, campaign_id: str) -> Path:
        return self.root / campaign_id

    def state_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / "campaign.json"

    def log_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / "campaign.log"

    def supervisor_log_path(self, campaign_id: str) -> Path:
        return self.campaign_dir(campaign_id) / "supervisor.log"

    def _atomic_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _creation_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".creation.lock"
        identity = {"pid": os.getpid(), "started_epoch": psutil.Process(os.getpid()).create_time()}
        for attempt in range(2):
            try:
                with lock_path.open("x", encoding="utf-8") as handle:
                    json.dump(identity, handle, sort_keys=True)
                break
            except FileExistsError:
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    existing = {}
                if self.process_matches(existing.get("pid"), existing.get("started_epoch")):
                    raise RuntimeError("another campaign creation is in progress")
                if attempt == 1:
                    raise RuntimeError("stale campaign creation lock could not be reclaimed")
                lock_path.unlink(missing_ok=True)
        try:
            yield
        finally:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
            if existing == identity:
                lock_path.unlink(missing_ok=True)

    def save(self, campaign: ValidationCampaign) -> ValidationCampaign:
        self._atomic_json(self.state_path(campaign.campaign_id), campaign_to_dict(campaign))
        return campaign

    def load(self, campaign_id: str) -> ValidationCampaign:
        value = json.loads(self.state_path(campaign_id).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("campaign state must be a JSON object")
        return campaign_from_dict(value)

    def active_id(self) -> str | None:
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
            campaign_id = value.get("campaign_id") if isinstance(value, dict) else None
            return campaign_id if isinstance(campaign_id, str) and campaign_id else None
        except (OSError, ValueError):
            return None

    def current(self) -> ValidationCampaign | None:
        campaign_id = self.active_id()
        if campaign_id is None:
            return None
        try:
            return self.load(campaign_id)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def set_active(self, campaign_id: str) -> None:
        self._atomic_json(self.active_path, {"campaign_id": campaign_id})

    def create(self, campaign: ValidationCampaign) -> ValidationCampaign:
        with self._creation_lock():
            for existing in self.list():
                if existing.state not in ACTIVE_CAMPAIGN_STATES:
                    continue
                if self.process_matches(existing.supervisor_pid, existing.supervisor_started_epoch):
                    raise RuntimeError(f"active validation campaign {existing.campaign_id} already owns supervision")
                interrupted = replace(existing, state=CampaignState.INTERRUPTED, updated_at=utc_now(),
                                      last_reason="campaign supervisor identity is stale")
                self.save(interrupted)
            directory = self.campaign_dir(campaign.campaign_id)
            directory.mkdir(parents=True, exist_ok=False)
            self.save(campaign)
            self.set_active(campaign.campaign_id)
            self.append_log(campaign.campaign_id, "campaign created")
        return campaign

    def append_log(self, campaign_id: str, message: str) -> None:
        with self.log_path(campaign_id).open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now().isoformat()} {message}\n")

    def list(self) -> tuple[ValidationCampaign, ...]:
        if not self.root.is_dir():
            return ()
        campaigns = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not (directory / "campaign.json").is_file():
                continue
            try:
                campaigns.append(self.load(directory.name))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return tuple(sorted(campaigns, key=lambda item: (item.created_at, item.campaign_id)))


def requested_suites(registry: SuiteRegistry, target: str) -> tuple[str, ...]:
    if target not in CAMPAIGN_SUITE_ORDER:
        raise ValueError(f"unknown campaign target: {target}")
    available = set(registry.names())
    through = CAMPAIGN_SUITE_ORDER[:CAMPAIGN_SUITE_ORDER.index(target) + 1]
    missing = [name for name in through if name not in available]
    if missing:
        raise ValueError(f"campaign target requires unregistered suites: {', '.join(missing)}")
    return tuple(through)


def _validate_duration(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def resource_probe() -> tuple[ResourceDecision, tuple[str, ...]]:
    manager = ResourceManager(HardwareTelemetryService(Path.cwd()))
    assessment = manager.assess(ResourceRequest("validation-campaign-preflight", 256 * MIB, 16 * MIB))
    return assessment.decision, assessment.reasons


def _blocked_results(result: ValidationRunResult) -> list[dict]:
    payload = result_to_dict(result)
    return [item for item in payload["scenario_results"] if item["status"] == ValidationStatus.BLOCKED.value]


def retryable_resource_delay(result: ValidationRunResult) -> bool:
    blocked = _blocked_results(result)
    return bool(blocked) and all(item["evidence"].get("resource_decision") == ResourceDecision.DELAY.value for item in blocked)


def campaign_report_text(campaign: ValidationCampaign) -> str:
    lines = ["T/TEA Validation Campaign", "==========================", f"Campaign: {campaign.campaign_id}",
             f"Target: {campaign.target_suite}", ""]
    for item in campaign.history:
        lines.append(f"{item.status.value:<8} {item.suite}  {item.run_id}")
        lines.append(f"         {item.reason}")
    if campaign.wait_periods:
        lines.append("")
        lines.append("Resource waits:")
        for item in campaign.wait_periods:
            finished = item.finished_at.isoformat() if item.finished_at is not None else "open"
            lines.append(f"  {item.started_at.isoformat()} -> {finished}: {item.reason}")
    cleanup_verified = campaign.state is not CampaignState.INTERRUPTED and (
        bool(campaign.history) or campaign.state in (CampaignState.CANCELLED, CampaignState.BLOCKED)
    ) and all(item.cleanup_verified for item in campaign.history)
    lines.extend(("", f"State: {campaign.state.value}", f"Retries: {campaign.retry_count}",
                  f"Cleanup verified: {'YES' if cleanup_verified else 'NO'}"))
    return "\n".join(lines) + "\n"


def write_campaign_report(store: CampaignStore, campaign: ValidationCampaign) -> None:
    directory = store.campaign_dir(campaign.campaign_id)
    payload = campaign_to_dict(campaign)
    payload["cleanup_verified"] = campaign.state is not CampaignState.INTERRUPTED and (
        bool(campaign.history) or campaign.state in (CampaignState.CANCELLED, CampaignState.BLOCKED)
    ) and all(item.cleanup_verified for item in campaign.history)
    store._atomic_json(directory / "campaign-report.json", payload)
    temporary = directory / f".campaign-report.txt.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(campaign_report_text(campaign), encoding="utf-8")
        temporary.replace(directory / "campaign-report.txt")
    finally:
        temporary.unlink(missing_ok=True)


class CampaignSupervisor:
    def __init__(self, campaign_id: str, store: CampaignStore, runner: ValidationRunner,
                 probe: Callable[[], tuple[ResourceDecision, tuple[str, ...]]] = resource_probe,
                 now: Callable[[], datetime] = utc_now, sleep: Callable[[float], None] = time.sleep) -> None:
        self.campaign_id, self.store, self.runner = campaign_id, store, runner
        self.probe, self.now, self.sleep = probe, now, sleep

    def _save(self, campaign: ValidationCampaign) -> ValidationCampaign:
        try:
            current = self.store.load(campaign.campaign_id)
            if current.cancel_requested and not campaign.cancel_requested:
                campaign = replace(campaign, cancel_requested=True)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return self.store.save(campaign)

    def _terminal(self, campaign: ValidationCampaign, state: CampaignState, reason: str) -> ValidationCampaign:
        current = self.now()
        wait_periods = campaign.wait_periods
        if wait_periods and wait_periods[-1].finished_at is None:
            wait_periods = wait_periods[:-1] + (replace(wait_periods[-1], finished_at=current),)
        campaign = replace(campaign, state=state, updated_at=current, heartbeat_at=current,
                           next_retry_at=None, last_reason=reason, wait_periods=wait_periods)
        self._save(campaign)
        self.store.append_log(campaign.campaign_id, f"campaign {state.value.lower()}: {reason}")
        write_campaign_report(self.store, campaign)
        return campaign

    def _cancel_if_requested(self, campaign: ValidationCampaign) -> ValidationCampaign | None:
        latest = self.store.load(campaign.campaign_id)
        if latest.cancel_requested:
            return self._terminal(latest, CampaignState.CANCELLED, "campaign cancellation requested")
        return None

    def _wait(self, campaign: ValidationCampaign) -> ValidationCampaign | None:
        while True:
            cancelled = self._cancel_if_requested(campaign)
            if cancelled is not None:
                return cancelled
            current = self.now()
            if current >= campaign.deadline_at:
                return self._terminal(campaign, CampaignState.BLOCKED, "campaign maximum duration elapsed")
            if campaign.next_retry_at is None or current >= campaign.next_retry_at:
                return None
            self.sleep(min(1.0, max(0.0, (campaign.next_retry_at - current).total_seconds())))

    def record_result(self, campaign: ValidationCampaign, result: ValidationRunResult) -> ValidationCampaign:
        reason = "; ".join(dict.fromkeys(reason for item in result.scenario_results for reason in item.reasons
                                         if item.status in (ValidationStatus.FAIL, ValidationStatus.BLOCKED)))
        if not reason:
            reason = f"validation suite {result.suite} {result.overall_status.value.lower()}"
        record = CampaignRunRecord(result.suite, result.run_id, result.overall_status, reason,
                                   result.started_at, result.finished_at, result.cleanup_verified)
        campaign = replace(campaign, history=campaign.history + (record,), last_validation_run_id=result.run_id,
                           last_validation_status=result.overall_status, last_reason=reason,
                           updated_at=self.now(), heartbeat_at=self.now())
        if result.interrupted:
            return self._terminal(campaign, CampaignState.INTERRUPTED, reason)
        if result.overall_status is ValidationStatus.FAIL:
            return self._terminal(campaign, CampaignState.FAILED, reason)
        if result.overall_status is ValidationStatus.BLOCKED:
            if retryable_resource_delay(result):
                next_retry = self.now() + timedelta(seconds=campaign.poll_interval_seconds)
                waits = campaign.wait_periods + (CampaignWaitPeriod(self.now(), None, reason),)
                waiting = replace(campaign, state=CampaignState.WAITING_FOR_RESOURCES,
                                  retry_count=campaign.retry_count + 1, next_retry_at=next_retry,
                                  wait_periods=waits)
                self.store.append_log(campaign.campaign_id, f"waiting for resources until {next_retry.isoformat()}: {reason}")
                return self._save(waiting)
            return self._terminal(campaign, CampaignState.BLOCKED, reason)
        index = campaign.requested_suites.index(campaign.current_suite)
        if index + 1 >= len(campaign.requested_suites):
            return self._terminal(campaign, CampaignState.COMPLETED, "all requested validation suites passed")
        next_suite = campaign.requested_suites[index + 1]
        return self._save(replace(campaign, current_suite=next_suite, state=CampaignState.RUNNING,
                                  updated_at=self.now(), heartbeat_at=self.now(), next_retry_at=None))

    def run(self) -> ValidationCampaign:
        campaign = self.store.load(self.campaign_id)
        try:
            while True:
                cancelled = self._cancel_if_requested(campaign)
                if cancelled is not None:
                    return cancelled
                if self.now() >= campaign.deadline_at:
                    return self._terminal(campaign, CampaignState.BLOCKED, "campaign maximum duration elapsed")
                if campaign.state is CampaignState.WAITING_FOR_RESOURCES:
                    terminal = self._wait(campaign)
                    if terminal is not None:
                        return terminal
                    try:
                        decision, reasons = self.probe()
                    except Exception as error_value:
                        return self._terminal(campaign, CampaignState.BLOCKED, f"resource probe failed: {error_value}")
                    if decision is ResourceDecision.DENY:
                        return self._terminal(campaign, CampaignState.BLOCKED, "; ".join(reasons))
                    if decision is ResourceDecision.DELAY:
                        current = self.now()
                        campaign = self._save(replace(campaign, retry_count=campaign.retry_count + 1,
                                                      updated_at=current, heartbeat_at=current,
                                                      next_retry_at=current + timedelta(seconds=campaign.poll_interval_seconds),
                                                      last_reason="; ".join(reasons)))
                        continue
                    current = self.now()
                    waits = campaign.wait_periods
                    if waits and waits[-1].finished_at is None:
                        waits = waits[:-1] + (replace(waits[-1], finished_at=current),)
                    campaign = self._save(replace(campaign, state=CampaignState.RUNNING, updated_at=current,
                                                  heartbeat_at=current, next_retry_at=None, wait_periods=waits))
                campaign = self._save(replace(campaign, state=CampaignState.RUNNING, updated_at=self.now(),
                                              heartbeat_at=self.now(), next_retry_at=None))
                result = self.runner.run(campaign.current_suite)
                campaign = self.record_result(campaign, result)
                if campaign.state in ACTIVE_CAMPAIGN_STATES:
                    continue
                return campaign
        except KeyboardInterrupt:
            return self._terminal(campaign, CampaignState.INTERRUPTED, "campaign supervisor interrupted")
        except BaseException as error_value:
            return self._terminal(campaign, CampaignState.FAILED,
                                  f"campaign supervisor failed: {type(error_value).__name__}: {error_value}")


class CampaignManager:
    def __init__(self, store: CampaignStore | None = None, registry: SuiteRegistry | None = None,
                 report_root: Path = DEFAULT_REPORT_ROOT, process_factory=subprocess.Popen,
                 process_api=psutil.Process, sleep: Callable[[float], None] = time.sleep,
                 benchmark_state_path: Path = DEFAULT_STATE_PATH) -> None:
        self.store = store or CampaignStore()
        self.registry = registry or build_registry()
        self.report_root = report_root
        self.process_factory, self.process_api, self.sleep = process_factory, process_api, sleep
        self.benchmark_state_path = benchmark_state_path

    def _spawn(self, campaign: ValidationCampaign):
        claim_token = uuid4().hex
        launching = replace(campaign, supervisor_claim_token=claim_token, updated_at=utc_now(),
                            last_reason="waiting for detached supervisor identity claim")
        self.store.save(launching)
        command = [sys.executable, "-m", "ttea.validation", "campaign", "_supervise", "--campaign-id", campaign.campaign_id,
                   "--claim-token", claim_token]
        kwargs = {"cwd": str(Path.cwd()), "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
                  "close_fds": True}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        with self.store.supervisor_log_path(campaign.campaign_id).open("a", encoding="utf-8") as log_handle:
            kwargs["stdout"] = log_handle
            process = self.process_factory(command, **kwargs)
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                claimed = self.store.load(campaign.campaign_id)
                if claimed.supervisor_claim_token is None and claimed.supervisor_pid != campaign.supervisor_pid:
                    if not self.store.process_matches(claimed.supervisor_pid, claimed.supervisor_started_epoch):
                        raise RuntimeError("detached supervisor claimed state with an unverifiable identity")
                    self.store.append_log(campaign.campaign_id, f"detached supervisor PID {claimed.supervisor_pid} recorded")
                    return claimed
                if claimed.state in (CampaignState.FAILED, CampaignState.INTERRUPTED):
                    raise RuntimeError(claimed.last_reason or "detached supervisor startup failed")
                self.sleep(0.05)
            raise RuntimeError("detached campaign supervisor did not claim ownership in time")
        except BaseException:
            latest = self.store.load(campaign.campaign_id)
            if latest.supervisor_claim_token is None and latest.supervisor_pid is not None:
                identity = OwnedProcessIdentity(latest.supervisor_pid, latest.supervisor_started_epoch)
                stop_verified_process(identity, timeout_seconds=3.0, process_api=self.process_api)
            else:
                try:
                    process.terminate()
                except Exception:
                    pass
            raise

    def start(self, target: str, poll_interval_seconds: float = 30.0,
              max_duration_seconds: float = 6 * 60 * 60, detach: bool = True) -> ValidationCampaign:
        _validate_duration(poll_interval_seconds, "poll_interval_seconds")
        _validate_duration(max_duration_seconds, "max_duration_seconds")
        suites = requested_suites(self.registry, target)
        now = utc_now()
        launcher = self.process_api(os.getpid())
        campaign = ValidationCampaign(
            campaign_id=f"campaign-{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}",
            requested_suites=suites, target_suite=target, current_suite=suites[0], state=CampaignState.CREATED,
            created_at=now, updated_at=now, deadline_at=now + timedelta(seconds=max_duration_seconds),
            supervisor_pid=os.getpid(), supervisor_started_epoch=launcher.create_time(), heartbeat_at=now,
            last_reason="campaign configuration validated", poll_interval_seconds=poll_interval_seconds,
        )
        self.store.create(campaign)
        if detach:
            try:
                return self._spawn(campaign)
            except BaseException as error_value:
                failed = replace(campaign, state=CampaignState.FAILED, updated_at=utc_now(),
                                 last_reason=f"detached supervisor startup failed: {error_value}")
                self.store.save(failed)
                write_campaign_report(self.store, failed)
                raise
        claimed = replace(campaign, supervisor_pid=os.getpid(), supervisor_started_epoch=launcher.create_time())
        self.store.save(claimed)
        return CampaignSupervisor(claimed.campaign_id, self.store,
                                  ValidationRunner(self.registry, self.report_root)).run()

    def status(self) -> tuple[ValidationCampaign | None, bool]:
        campaign = self.store.current()
        if campaign is None:
            return None, False
        alive = campaign.state in ACTIVE_CAMPAIGN_STATES and self.store.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch)
        if campaign.state in ACTIVE_CAMPAIGN_STATES and not alive:
            campaign = replace(campaign, state=CampaignState.INTERRUPTED, updated_at=utc_now(),
                               last_reason="campaign supervisor identity is stale", next_retry_at=None)
            self.store.save(campaign)
            write_campaign_report(self.store, campaign)
        return campaign, alive

    def cancel(self, graceful_timeout_seconds: float = 15.0) -> ValidationCampaign:
        campaign, alive = self.status()
        if campaign is None:
            raise RuntimeError("no validation campaign exists")
        if campaign.state not in ACTIVE_CAMPAIGN_STATES:
            if campaign.state is CampaignState.INTERRUPTED:
                campaign = replace(campaign, state=CampaignState.CANCELLED, updated_at=utc_now(),
                                   cancel_requested=True, last_reason="stale campaign cancelled")
                self.store.save(campaign)
                write_campaign_report(self.store, campaign)
            return campaign
        campaign = replace(campaign, cancel_requested=True, updated_at=utc_now(), last_reason="cancellation requested")
        self.store.save(campaign)
        deadline = time.monotonic() + graceful_timeout_seconds
        while alive and time.monotonic() < deadline:
            self.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            alive = self.store.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch)
        if alive:
            identity = OwnedProcessIdentity(campaign.supervisor_pid, campaign.supervisor_started_epoch)
            stop_verified_process(identity, timeout_seconds=3.0, process_api=self.process_api)
        latest = self.store.load(campaign.campaign_id)
        if latest.state in ACTIVE_CAMPAIGN_STATES:
            benchmark_state = BenchmarkStateStore(self.benchmark_state_path)
            state = benchmark_state.snapshot() or {}
            owned_remains = any(benchmark_state.process_matches(state.get(f"{kind}_pid"), state.get(f"{kind}_started_epoch"))
                                for kind in ("worker", "server"))
            terminal_state = CampaignState.INTERRUPTED if owned_remains else CampaignState.CANCELLED
            reason = ("campaign supervisor stopped; live benchmark ownership state was preserved"
                      if owned_remains else "campaign cancelled after verified supervisor stop")
            latest = replace(latest, state=terminal_state, updated_at=utc_now(), cancel_requested=True,
                             next_retry_at=None, last_reason=reason)
            self.store.save(latest)
            write_campaign_report(self.store, latest)
        return latest

    def resume(self, graceful_duration_seconds: float | None = None) -> ValidationCampaign:
        campaign, alive = self.status()
        if campaign is None:
            raise RuntimeError("no validation campaign exists")
        if alive:
            raise RuntimeError("active validation campaign cannot be resumed twice")
        if campaign.state not in (CampaignState.INTERRUPTED, CampaignState.BLOCKED):
            raise RuntimeError(f"campaign in state {campaign.state.value} is not resumable")
        guard = BenchmarkStateStore(self.benchmark_state_path)
        guard.acquire()
        guard.release()
        duration = (campaign.deadline_at - campaign.created_at).total_seconds() if graceful_duration_seconds is None else graceful_duration_seconds
        _validate_duration(duration, "max_duration_seconds")
        now = utc_now()
        resumed = replace(campaign, state=CampaignState.CREATED, updated_at=now, deadline_at=now + timedelta(seconds=duration),
                          supervisor_pid=os.getpid(), supervisor_started_epoch=self.process_api(os.getpid()).create_time(),
                          heartbeat_at=now, next_retry_at=None, cancel_requested=False, supervisor_claim_token=None,
                          last_reason="campaign resume validated")
        self.store.save(resumed)
        return self._spawn(resumed)


def supervise_campaign(campaign_id: str, claim_token: str, store: CampaignStore | None = None,
                       registry: SuiteRegistry | None = None) -> ValidationCampaign:
    store = store or CampaignStore()
    registry = registry or build_registry()
    try:
        own_started = psutil.Process(os.getpid()).create_time()
        campaign = store.load(campaign_id)
        if campaign.state is not CampaignState.CREATED or campaign.supervisor_claim_token != claim_token:
            raise RuntimeError("detached campaign supervisor claim token is invalid or already used")
        campaign = replace(campaign, supervisor_pid=os.getpid(), supervisor_started_epoch=own_started,
                           supervisor_claim_token=None, updated_at=utc_now(), heartbeat_at=utc_now(),
                           last_reason="detached campaign supervisor started")
        store.save(campaign)
        store.append_log(campaign_id, "detached supervisor identity verified and claimed")
        return CampaignSupervisor(campaign_id, store, ValidationRunner(registry, DEFAULT_REPORT_ROOT)).run()
    except BaseException as error_value:
        try:
            campaign = store.load(campaign_id)
            if campaign.supervisor_pid == os.getpid() or campaign.supervisor_claim_token == claim_token:
                state = CampaignState.INTERRUPTED if isinstance(error_value, KeyboardInterrupt) else CampaignState.FAILED
                failed = replace(campaign, state=state, updated_at=utc_now(), heartbeat_at=utc_now(),
                                 last_reason=f"detached supervisor startup failed: {type(error_value).__name__}: {error_value}")
                store.save(failed)
                store.append_log(campaign_id, failed.last_reason)
                write_campaign_report(store, failed)
        except Exception:
            pass
        raise


def campaign_status_text(campaign: ValidationCampaign, alive: bool) -> str:
    elapsed = max(0.0, (utc_now() - campaign.created_at).total_seconds())
    return "\n".join((
        f"Campaign: {campaign.campaign_id}", f"State: {campaign.state.value}",
        f"Current suite: {campaign.current_suite or '(none)'}",
        f"Last validation: {campaign.last_validation_status.value if campaign.last_validation_status else '(none)'}",
        f"Reason: {campaign.last_reason or '(none)'}",
        f"Next retry: {campaign.next_retry_at.isoformat() if campaign.next_retry_at else '(none)'}",
        f"Supervisor: {'alive' if alive else 'not running'}", f"Elapsed: {elapsed:.1f} seconds",
    )) + "\n"
