"""Immutable persistent models for autonomous validation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any

from .models import ValidationStatus


class CampaignState(str, Enum):
    CREATED = "CREATED"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


ACTIVE_CAMPAIGN_STATES = frozenset((CampaignState.CREATED, CampaignState.WAITING_FOR_RESOURCES, CampaignState.RUNNING))
TERMINAL_CAMPAIGN_STATES = frozenset((CampaignState.BLOCKED, CampaignState.COMPLETED, CampaignState.FAILED,
                                      CampaignState.CANCELLED, CampaignState.INTERRUPTED))


def _utc(value: datetime | None, name: str) -> None:
    if value is not None and (not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value)):
        raise ValueError(f"{name} must be timezone-aware UTC when present")


def _text(value: str | None, name: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CampaignRunRecord:
    suite: str
    run_id: str
    status: ValidationStatus
    reason: str
    started_at: datetime
    finished_at: datetime
    cleanup_verified: bool

    def __post_init__(self) -> None:
        _text(self.suite, "suite")
        _text(self.run_id, "run_id")
        _text(self.reason, "reason")
        if not isinstance(self.status, ValidationStatus):
            raise ValueError("status must be a ValidationStatus")
        _utc(self.started_at, "started_at")
        _utc(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if not isinstance(self.cleanup_verified, bool):
            raise ValueError("cleanup_verified must be a bool")


@dataclass(frozen=True)
class CampaignWaitPeriod:
    started_at: datetime
    finished_at: datetime | None
    reason: str

    def __post_init__(self) -> None:
        _utc(self.started_at, "started_at")
        _utc(self.finished_at, "finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        _text(self.reason, "reason")


@dataclass(frozen=True)
class ValidationCampaign:
    campaign_id: str
    requested_suites: tuple[str, ...]
    target_suite: str
    current_suite: str | None
    state: CampaignState
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime
    supervisor_pid: int | None = None
    supervisor_started_epoch: float | None = None
    heartbeat_at: datetime | None = None
    last_validation_run_id: str | None = None
    last_validation_status: ValidationStatus | None = None
    last_reason: str | None = None
    retry_count: int = 0
    next_retry_at: datetime | None = None
    poll_interval_seconds: float = 30.0
    cancel_requested: bool = False
    supervisor_claim_token: str | None = None
    history: tuple[CampaignRunRecord, ...] = ()
    wait_periods: tuple[CampaignWaitPeriod, ...] = ()

    def __post_init__(self) -> None:
        _text(self.campaign_id, "campaign_id")
        if self.campaign_id in (".", "..") or "/" in self.campaign_id or "\\" in self.campaign_id:
            raise ValueError("campaign_id must be a single safe path component")
        if not isinstance(self.requested_suites, tuple) or not self.requested_suites or any(not isinstance(item, str) or not item.strip() for item in self.requested_suites):
            raise ValueError("requested_suites must be a non-empty tuple of suite names")
        if len(set(self.requested_suites)) != len(self.requested_suites):
            raise ValueError("requested_suites must not contain duplicates")
        _text(self.target_suite, "target_suite")
        if self.target_suite != self.requested_suites[-1]:
            raise ValueError("target_suite must be the last requested suite")
        _text(self.current_suite, "current_suite", optional=True)
        if self.current_suite is not None and self.current_suite not in self.requested_suites:
            raise ValueError("current_suite must be requested")
        if not isinstance(self.state, CampaignState):
            raise ValueError("state must be a CampaignState")
        for name in ("created_at", "updated_at", "deadline_at", "heartbeat_at", "next_retry_at"):
            _utc(getattr(self, name), name)
        if self.updated_at < self.created_at or self.deadline_at <= self.created_at:
            raise ValueError("campaign timestamps are inconsistent")
        if self.supervisor_pid is None:
            if self.supervisor_started_epoch is not None:
                raise ValueError("supervisor creation time requires a PID")
        elif not isinstance(self.supervisor_pid, int) or isinstance(self.supervisor_pid, bool) or self.supervisor_pid <= 0:
            raise ValueError("supervisor_pid must be a positive integer when present")
        if self.supervisor_started_epoch is not None and (not isinstance(self.supervisor_started_epoch, (int, float)) or isinstance(self.supervisor_started_epoch, bool) or not math.isfinite(self.supervisor_started_epoch) or self.supervisor_started_epoch < 0):
            raise ValueError("supervisor_started_epoch must be finite and non-negative")
        _text(self.last_validation_run_id, "last_validation_run_id", optional=True)
        if self.last_validation_status is not None and not isinstance(self.last_validation_status, ValidationStatus):
            raise ValueError("last_validation_status must be a ValidationStatus when present")
        _text(self.last_reason, "last_reason", optional=True)
        if not isinstance(self.retry_count, int) or isinstance(self.retry_count, bool) or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if not isinstance(self.poll_interval_seconds, (int, float)) or isinstance(self.poll_interval_seconds, bool) or not math.isfinite(self.poll_interval_seconds) or self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be a finite positive number")
        if not isinstance(self.cancel_requested, bool):
            raise ValueError("cancel_requested must be a bool")
        _text(self.supervisor_claim_token, "supervisor_claim_token", optional=True)
        if not isinstance(self.history, tuple) or any(not isinstance(item, CampaignRunRecord) for item in self.history):
            raise ValueError("history must be a tuple of CampaignRunRecord values")
        if not isinstance(self.wait_periods, tuple) or any(not isinstance(item, CampaignWaitPeriod) for item in self.wait_periods):
            raise ValueError("wait_periods must be a tuple of CampaignWaitPeriod values")


def campaign_to_dict(campaign: ValidationCampaign) -> dict[str, Any]:
    return {
        "campaign_id": campaign.campaign_id,
        "requested_suites": list(campaign.requested_suites),
        "target_suite": campaign.target_suite,
        "current_suite": campaign.current_suite,
        "state": campaign.state.value,
        "created_at": campaign.created_at.isoformat(),
        "updated_at": campaign.updated_at.isoformat(),
        "deadline_at": campaign.deadline_at.isoformat(),
        "supervisor_pid": campaign.supervisor_pid,
        "supervisor_started_epoch": campaign.supervisor_started_epoch,
        "heartbeat_at": None if campaign.heartbeat_at is None else campaign.heartbeat_at.isoformat(),
        "last_validation_run_id": campaign.last_validation_run_id,
        "last_validation_status": None if campaign.last_validation_status is None else campaign.last_validation_status.value,
        "last_reason": campaign.last_reason,
        "retry_count": campaign.retry_count,
        "next_retry_at": None if campaign.next_retry_at is None else campaign.next_retry_at.isoformat(),
        "poll_interval_seconds": campaign.poll_interval_seconds,
        "cancel_requested": campaign.cancel_requested,
        "supervisor_claim_token": campaign.supervisor_claim_token,
        "history": [
            {"suite": item.suite, "run_id": item.run_id, "status": item.status.value, "reason": item.reason,
             "started_at": item.started_at.isoformat(), "finished_at": item.finished_at.isoformat(),
             "cleanup_verified": item.cleanup_verified}
            for item in campaign.history
        ],
        "wait_periods": [
            {"started_at": item.started_at.isoformat(),
             "finished_at": None if item.finished_at is None else item.finished_at.isoformat(),
             "reason": item.reason}
            for item in campaign.wait_periods
        ],
    }


def campaign_from_dict(value: dict[str, Any]) -> ValidationCampaign:
    def timestamp(name: str) -> datetime | None:
        raw = value.get(name)
        return None if raw is None else datetime.fromisoformat(raw)

    history = tuple(CampaignRunRecord(
        suite=item["suite"], run_id=item["run_id"], status=ValidationStatus(item["status"]), reason=item["reason"],
        started_at=datetime.fromisoformat(item["started_at"]), finished_at=datetime.fromisoformat(item["finished_at"]),
        cleanup_verified=item["cleanup_verified"],
    ) for item in value.get("history", ()))
    wait_periods = tuple(CampaignWaitPeriod(
        started_at=datetime.fromisoformat(item["started_at"]),
        finished_at=None if item.get("finished_at") is None else datetime.fromisoformat(item["finished_at"]),
        reason=item["reason"],
    ) for item in value.get("wait_periods", ()))
    last_status = value.get("last_validation_status")
    return ValidationCampaign(
        campaign_id=value["campaign_id"], requested_suites=tuple(value["requested_suites"]), target_suite=value["target_suite"],
        current_suite=value.get("current_suite"), state=CampaignState(value["state"]), created_at=timestamp("created_at"),
        updated_at=timestamp("updated_at"), deadline_at=timestamp("deadline_at"), supervisor_pid=value.get("supervisor_pid"),
        supervisor_started_epoch=value.get("supervisor_started_epoch"), heartbeat_at=timestamp("heartbeat_at"),
        last_validation_run_id=value.get("last_validation_run_id"),
        last_validation_status=None if last_status is None else ValidationStatus(last_status), last_reason=value.get("last_reason"),
        retry_count=value.get("retry_count", 0), next_retry_at=timestamp("next_retry_at"),
        poll_interval_seconds=value.get("poll_interval_seconds", 30.0), cancel_requested=value.get("cancel_requested", False),
        supervisor_claim_token=value.get("supervisor_claim_token"), history=history, wait_periods=wait_periods,
    )
