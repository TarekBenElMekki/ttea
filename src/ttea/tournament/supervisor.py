"""Detached persistent supervision for the controlled survival tournament."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
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
from ttea.resources import ResourceManager
from ttea.telemetry import HardwareTelemetryService
from ttea.validation.suites import stop_verified_process

from .catalog import build_survival_manifest
from .execution import (
    ACTIVE_SURVIVAL_STATES, CandidateSurvivalRecord, SurvivalCampaign,
    SurvivalCampaignState, SurvivalExecutionPolicy, SurvivalOutcome,
    SurvivalStore, utc_now,
)
from .ollama_provider import OllamaTournamentProvider
from .survival import SurvivalTournamentEngine, write_survival_report


DEFAULT_TOURNAMENT_ROOT = Path("data/tournaments")
DEFAULT_BENCHMARK_STATE = Path("data/runtime/benchmark.json")


class TournamentSupervisorManager:
    def __init__(self, store: SurvivalStore | None = None, *, process_factory=subprocess.Popen,
                 process_api=psutil.Process, sleep: Callable[[float], None] = time.sleep,
                 benchmark_state_path: Path = DEFAULT_BENCHMARK_STATE,
                 provider_factory=OllamaTournamentProvider,
                 engine_factory=SurvivalTournamentEngine) -> None:
        self.store = store or SurvivalStore(DEFAULT_TOURNAMENT_ROOT)
        self.process_factory, self.process_api, self.sleep = process_factory, process_api, sleep
        self.benchmark_state_path = Path(benchmark_state_path)
        self.provider_factory, self.engine_factory = provider_factory, engine_factory

    @staticmethod
    def process_matches(pid: object, started_epoch: object) -> bool:
        return BenchmarkStateStore.process_matches(pid, started_epoch)

    def _spawn(self, campaign: SurvivalCampaign) -> SurvivalCampaign:
        claim = uuid4().hex
        launching = replace(campaign, supervisor_claim_token=claim, updated_at=utc_now(),
                            last_reason="waiting for detached supervisor ownership claim")
        self.store.save(launching)
        command = [sys.executable, "-m", "ttea.tournament", "--root", str(self.store.root),
                   "_supervise", "--tournament-id", campaign.tournament_id, "--claim-token", claim]
        kwargs = {"cwd": str(Path.cwd()), "stdin": subprocess.DEVNULL,
                  "stderr": subprocess.STDOUT, "close_fds": True}
        if os.name == "nt":
            kwargs["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                                       subprocess.CREATE_NO_WINDOW)
        else:
            kwargs["start_new_session"] = True
        log_path = self.store.supervisor_log_path(campaign.tournament_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            kwargs["stdout"] = log_handle
            process = self.process_factory(command, **kwargs)
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                claimed = self.store.load(campaign.tournament_id)
                if claimed.supervisor_claim_token is None and claimed.supervisor_pid != campaign.supervisor_pid:
                    if not self.process_matches(claimed.supervisor_pid, claimed.supervisor_started_epoch):
                        raise RuntimeError("detached tournament supervisor identity is unverifiable")
                    return claimed
                if claimed.state in (SurvivalCampaignState.FAILED, SurvivalCampaignState.INTERRUPTED):
                    raise RuntimeError(claimed.last_reason or "detached supervisor startup failed")
                self.sleep(0.05)
            raise RuntimeError("detached tournament supervisor did not claim ownership in time")
        except BaseException:
            latest = self.store.load(campaign.tournament_id)
            if latest.supervisor_claim_token is None and latest.supervisor_pid is not None:
                stop_verified_process(OwnedProcessIdentity(latest.supervisor_pid, latest.supervisor_started_epoch),
                                      timeout_seconds=3.0, process_api=self.process_api)
            else:
                try:
                    process.terminate()
                except Exception:
                    pass
            raise

    def start(self, through: str = "survival", allow_model_download: bool = False,
              detach: bool = True, policy: SurvivalExecutionPolicy | None = None) -> SurvivalCampaign:
        if through != "survival":
            raise ValueError("only --through survival is currently executable")
        if not isinstance(allow_model_download, bool):
            raise ValueError("allow_model_download must be bool")
        active = self.store.active()
        if active is not None and self.process_matches(active.supervisor_pid, active.supervisor_started_epoch):
            raise RuntimeError(f"active tournament {active.tournament_id} already owns supervision")
        if active is not None:
            stale_records = {
                key: (replace(value, outcome=SurvivalOutcome.INTERRUPTED,
                              reason="stale running attempt recovered")
                      if value.outcome is SurvivalOutcome.RUNNING else value)
                for key, value in active.records.items()
            }
            self.store.save(replace(active, state=SurvivalCampaignState.INTERRUPTED,
                                    records=stale_records, updated_at=utc_now(),
                                    last_event="stale supervisor detected",
                                    last_reason="previous supervisor identity is stale"))
        policy = policy or SurvivalExecutionPolicy(allow_model_download=allow_model_download)
        if policy.allow_model_download != allow_model_download:
            raise ValueError("download authorization flag must match persisted execution policy")
        manifest = build_survival_manifest()
        now = utc_now()
        tournament_id = f"tournament-{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
        identity = self.process_api(os.getpid())
        campaign = SurvivalCampaign(
            tournament_id, manifest.digest, SurvivalCampaignState.CREATED, now, now,
            now + timedelta(seconds=policy.resource_wait_timeout_seconds),
            {candidate.candidate_id: CandidateSurvivalRecord(candidate.candidate_id, candidate.variant.model)
             for candidate in manifest.candidates},
            supervisor_pid=os.getpid(), supervisor_started_epoch=identity.create_time(),
            heartbeat_at=now, last_event="tournament created",
            last_reason="controlled survival policy persisted",
        )
        self.store.create(campaign, manifest, policy)
        if detach:
            return self._spawn(campaign)
        return self._run_claimed(campaign)

    def _run_claimed(self, campaign: SurvivalCampaign) -> SurvivalCampaign:
        policy = self.store.load_policy(campaign.tournament_id)
        manifest = self.store.tournaments.load_manifest(campaign.tournament_id)
        manager = ResourceManager(HardwareTelemetryService(Path.cwd()))
        provider = self.provider_factory(manifest, manager, policy, state_path=self.benchmark_state_path)
        return self.engine_factory(campaign.tournament_id, self.store, provider).run()

    def status(self, tournament_id: str | None = None) -> tuple[SurvivalCampaign | None, bool]:
        campaign = self.store.load(tournament_id) if tournament_id else self.store.latest()
        if campaign is None:
            return None, False
        alive = campaign.state in ACTIVE_SURVIVAL_STATES and self.process_matches(
            campaign.supervisor_pid, campaign.supervisor_started_epoch)
        if campaign.state in ACTIVE_SURVIVAL_STATES and not alive:
            records = {
                key: (replace(value, outcome=SurvivalOutcome.INTERRUPTED,
                              reason="stale running attempt recovered")
                      if value.outcome is SurvivalOutcome.RUNNING else value)
                for key, value in campaign.records.items()
            }
            campaign = replace(campaign, state=SurvivalCampaignState.INTERRUPTED,
                               records=records, updated_at=utc_now(), heartbeat_at=utc_now(),
                               next_retry_at=None, last_event="stale supervisor detected",
                               last_reason="supervisor identity is stale")
            self.store.save(campaign)
            write_survival_report(self.store, campaign, self.store.tournaments.load_manifest(campaign.tournament_id))
        return campaign, alive

    def resume(self, tournament_id: str | None = None) -> SurvivalCampaign:
        campaign, alive = self.status(tournament_id)
        if campaign is None:
            raise RuntimeError("no survival tournament exists")
        if alive:
            raise RuntimeError("active tournament cannot be resumed twice")
        if campaign.state is not SurvivalCampaignState.INTERRUPTED:
            raise RuntimeError(f"tournament in state {campaign.state.value} is not resumable")
        # This acquire/release pair reclaims only stale ownership. Any verified
        # live worker/server identity blocks resume and is never guessed away.
        guard = BenchmarkStateStore(self.benchmark_state_path)
        guard.acquire()
        guard.release()
        now = utc_now()
        policy = self.store.load_policy(campaign.tournament_id)
        records = {
            key: (replace(value, outcome=SurvivalOutcome.PENDING)
                  if value.outcome is SurvivalOutcome.INTERRUPTED else value)
            for key, value in campaign.records.items()
        }
        resumed = replace(
            campaign, state=SurvivalCampaignState.CREATED, records=records,
            updated_at=now, deadline_at=now + timedelta(seconds=policy.resource_wait_timeout_seconds),
            cancellation_requested=False, supervisor_pid=os.getpid(),
            supervisor_started_epoch=self.process_api(os.getpid()).create_time(),
            supervisor_claim_token=None, heartbeat_at=now, next_retry_at=None,
            last_event="tournament resumed", last_reason="stale runtime ownership was cleared",
        )
        self.store.save(resumed, allow_resume=True)
        return self._spawn(resumed)

    def cancel(self, tournament_id: str | None = None,
               graceful_timeout_seconds: float = 15.0) -> SurvivalCampaign:
        campaign, alive = self.status(tournament_id)
        if campaign is None:
            raise RuntimeError("no survival tournament exists")
        if campaign.state not in ACTIVE_SURVIVAL_STATES:
            if campaign.state is SurvivalCampaignState.INTERRUPTED:
                cancelled = replace(campaign, state=SurvivalCampaignState.CANCELLED,
                                    cancellation_requested=True, updated_at=utc_now(),
                                    last_event="cancelled", last_reason="stale tournament cancelled")
                self.store.save(cancelled)
                return cancelled
            return campaign
        requested = replace(campaign, cancellation_requested=True, updated_at=utc_now(),
                            last_event="cancellation requested", last_reason="graceful cancellation requested")
        self.store.save(requested)
        deadline = time.monotonic() + graceful_timeout_seconds
        while alive and time.monotonic() < deadline:
            self.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            alive = self.process_matches(campaign.supervisor_pid, campaign.supervisor_started_epoch)
        if alive:
            stop_verified_process(OwnedProcessIdentity(campaign.supervisor_pid, campaign.supervisor_started_epoch),
                                  timeout_seconds=3.0, process_api=self.process_api)
        ownership = BenchmarkStateStore(self.benchmark_state_path)
        state = ownership.snapshot() or {}
        for kind in ("worker", "server"):
            pid, started = state.get(f"{kind}_pid"), state.get(f"{kind}_started_epoch")
            if ownership.process_matches(pid, started):
                stop_verified_process(OwnedProcessIdentity(pid, started), timeout_seconds=3.0,
                                      process_api=self.process_api)
        remaining = ownership.snapshot() or {}
        owned_remains = any(ownership.process_matches(remaining.get(f"{kind}_pid"), remaining.get(f"{kind}_started_epoch"))
                            for kind in ("worker", "server"))
        latest = self.store.load(campaign.tournament_id)
        if latest.state in ACTIVE_SURVIVAL_STATES:
            terminal = SurvivalCampaignState.INTERRUPTED if owned_remains else SurvivalCampaignState.CANCELLED
            records = dict(latest.records)
            if latest.current_candidate_id and records[latest.current_candidate_id].outcome is SurvivalOutcome.RUNNING:
                records[latest.current_candidate_id] = replace(
                    records[latest.current_candidate_id],
                    outcome=SurvivalOutcome.INTERRUPTED if owned_remains else SurvivalOutcome.CANCELLED,
                    reason="cancellation interrupted owned workload",
                )
            latest = replace(latest, state=terminal, records=records,
                             updated_at=utc_now(), cancellation_requested=True,
                             next_retry_at=None, last_event="cancelled" if not owned_remains else "interrupted",
                             last_reason=("owned runtime cleanup could not be verified" if owned_remains
                                          else "verified owned processes stopped"))
            self.store.save(latest)
            write_survival_report(self.store, latest, self.store.tournaments.load_manifest(latest.tournament_id))
        return latest


def supervise_tournament(tournament_id: str, claim_token: str,
                         store: SurvivalStore | None = None) -> SurvivalCampaign:
    store = store or SurvivalStore(DEFAULT_TOURNAMENT_ROOT)
    try:
        campaign = store.load(tournament_id)
        own_started = psutil.Process(os.getpid()).create_time()
        if campaign.state is not SurvivalCampaignState.CREATED or campaign.supervisor_claim_token != claim_token:
            raise RuntimeError("detached tournament supervisor claim token is invalid or already used")
        claimed = replace(campaign, supervisor_pid=os.getpid(), supervisor_started_epoch=own_started,
                          supervisor_claim_token=None, updated_at=utc_now(), heartbeat_at=utc_now(),
                          last_event="detached supervisor claimed",
                          last_reason="verified detached process identity")
        store.save(claimed)
        store.append_event(tournament_id, "detached supervisor claimed", utc_now())
        return TournamentSupervisorManager(store)._run_claimed(claimed)
    except BaseException as exc:
        try:
            campaign = store.load(tournament_id)
            if campaign.supervisor_pid == os.getpid() or campaign.supervisor_claim_token == claim_token:
                failed = replace(campaign, state=SurvivalCampaignState.INTERRUPTED,
                                 updated_at=utc_now(), heartbeat_at=utc_now(),
                                 last_event="supervisor interrupted",
                                 last_reason=f"{type(exc).__name__}: {exc}")
                store.save(failed)
        except Exception:
            pass
        raise

