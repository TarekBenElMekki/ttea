"""Authoritative in-memory lifecycle mutation APIs for T/TEA core models."""

from __future__ import annotations

from datetime import datetime, timezone
import logging

from ttea.logging import get_logger, setup_logging

from .enums import AgentStatus, ApprovalStatus, MissionStatus, TaskStatus
from .models import AgentSpec, Approval, Event, Mission, Task
from .transitions import validate_transition

LOGGER = get_logger("core.state_service")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _status_payload(entity_id: str, previous_status: str, new_status: str, reason: str | None) -> dict[str, str]:
    payload = {
        "entity_id": entity_id,
        "previous_status": previous_status,
        "new_status": new_status,
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def _ensure_logging() -> None:
    """Ensure the shared persistent logger is ready before mutation."""
    if not logging.getLogger("ttea").handlers:
        setup_logging()


def _log_status_change(entity_name: str, entity_id: str, previous_status: str, target_status: str) -> None:
    """Emit a concise informational status-change record."""
    LOGGER.info("%s %s status changed: %s -> %s", entity_name, entity_id, previous_status, target_status)


def transition_mission(mission: Mission, target: MissionStatus, reason: str | None = None) -> Event:
    """Validate, mutate, and describe a mission status change."""
    previous_status = mission.status
    validate_transition(previous_status, target)
    _ensure_logging()
    timestamp = _utc_now()
    event = Event(
        event_type="mission.status_changed",
        mission_id=mission.id,
        timestamp=timestamp,
        payload=_status_payload(mission.id, previous_status.value, target.value, reason),
    )
    mission.status = target
    mission.updated_at = timestamp
    _log_status_change("Mission", mission.id, previous_status.value, target.value)
    return event


def transition_task(task: Task, target: TaskStatus, reason: str | None = None) -> Event:
    """Validate, mutate, and describe a task status change."""
    previous_status = task.status
    validate_transition(previous_status, target)
    _ensure_logging()
    timestamp = _utc_now()
    event = Event(
        event_type="task.status_changed",
        mission_id=task.mission_id,
        task_id=task.id,
        timestamp=timestamp,
        payload=_status_payload(task.id, previous_status.value, target.value, reason),
    )
    task.status = target
    task.updated_at = timestamp
    _log_status_change("Task", task.id, previous_status.value, target.value)
    return event


def transition_agent(agent: AgentSpec, target: AgentStatus, reason: str | None = None) -> Event:
    """Validate, mutate, and describe an agent specification status change."""
    previous_status = agent.status
    validate_transition(previous_status, target)
    _ensure_logging()
    timestamp = _utc_now()
    event = Event(
        event_type="agent.status_changed",
        mission_id=agent.mission_id,
        agent_id=agent.id,
        timestamp=timestamp,
        payload=_status_payload(agent.id, previous_status.value, target.value, reason),
    )
    agent.status = target
    agent.updated_at = timestamp
    _log_status_change("Agent", agent.id, previous_status.value, target.value)
    return event


def resolve_approval(approval: Approval, target: ApprovalStatus, reason: str | None = None) -> Event:
    """Resolve a pending approval and return the corresponding core event."""
    previous_status = approval.status
    validate_transition(previous_status, target)
    _ensure_logging()
    timestamp = _utc_now()
    payload = _status_payload(approval.id, previous_status.value, target.value, reason)
    payload["approval_id"] = approval.id
    event = Event(
        event_type="approval.resolved",
        mission_id=approval.mission_id,
        timestamp=timestamp,
        payload=payload,
    )
    approval.status = target
    approval.resolved_at = timestamp
    _log_status_change("Approval", approval.id, previous_status.value, target.value)
    return event
