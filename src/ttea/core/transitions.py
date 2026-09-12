"""Deterministic lifecycle transition rules without state mutation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypeVar

from .enums import AgentStatus, ApprovalStatus, MissionStatus, TaskStatus
from .exceptions import InvalidStateTransition

StatusType = TypeVar("StatusType", bound=Enum)

MISSION_TRANSITIONS: Mapping[MissionStatus, frozenset[MissionStatus]] = {
    MissionStatus.CREATED: frozenset({MissionStatus.ANALYZING, MissionStatus.CANCELLED, MissionStatus.FAILED}),
    MissionStatus.ANALYZING: frozenset({MissionStatus.PLANNING, MissionStatus.BLOCKED, MissionStatus.PAUSED, MissionStatus.FAILED, MissionStatus.CANCELLED}),
    MissionStatus.PLANNING: frozenset({MissionStatus.AWAITING_APPROVAL, MissionStatus.BLOCKED, MissionStatus.PAUSED, MissionStatus.FAILED, MissionStatus.CANCELLED}),
    MissionStatus.AWAITING_APPROVAL: frozenset({MissionStatus.PLANNING, MissionStatus.READY, MissionStatus.BLOCKED, MissionStatus.CANCELLED}),
    MissionStatus.READY: frozenset({MissionStatus.RUNNING, MissionStatus.BLOCKED, MissionStatus.PAUSED, MissionStatus.FAILED, MissionStatus.CANCELLED}),
    MissionStatus.RUNNING: frozenset({MissionStatus.PAUSED, MissionStatus.BLOCKED, MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED}),
    MissionStatus.PAUSED: frozenset({MissionStatus.READY, MissionStatus.RUNNING, MissionStatus.BLOCKED, MissionStatus.FAILED, MissionStatus.CANCELLED}),
    MissionStatus.BLOCKED: frozenset({MissionStatus.READY, MissionStatus.PAUSED, MissionStatus.FAILED, MissionStatus.CANCELLED}),
    MissionStatus.COMPLETED: frozenset(),
    MissionStatus.FAILED: frozenset(),
    MissionStatus.CANCELLED: frozenset(),
}

TASK_TRANSITIONS: Mapping[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.BLOCKED: frozenset({TaskStatus.PENDING, TaskStatus.READY, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.READY: frozenset({TaskStatus.ASSIGNED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.ASSIGNED: frozenset({TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.VERIFYING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.VERIFYING: frozenset({TaskStatus.READY, TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset({TaskStatus.READY, TaskStatus.CANCELLED}),
    TaskStatus.CANCELLED: frozenset(),
}

AGENT_TRANSITIONS: Mapping[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.PROPOSED: frozenset({AgentStatus.APPROVED, AgentStatus.TERMINATED}),
    AgentStatus.APPROVED: frozenset({AgentStatus.CREATED, AgentStatus.TERMINATED}),
    AgentStatus.CREATED: frozenset({AgentStatus.IDLE, AgentStatus.FAILED, AgentStatus.TERMINATED}),
    AgentStatus.IDLE: frozenset({AgentStatus.WORKING, AgentStatus.BLOCKED, AgentStatus.PAUSED, AgentStatus.COMPLETED, AgentStatus.TERMINATED}),
    AgentStatus.WORKING: frozenset({AgentStatus.IDLE, AgentStatus.BLOCKED, AgentStatus.PAUSED, AgentStatus.FAILED, AgentStatus.COMPLETED, AgentStatus.TERMINATED}),
    AgentStatus.BLOCKED: frozenset({AgentStatus.IDLE, AgentStatus.WORKING, AgentStatus.PAUSED, AgentStatus.FAILED, AgentStatus.TERMINATED}),
    AgentStatus.PAUSED: frozenset({AgentStatus.IDLE, AgentStatus.WORKING, AgentStatus.BLOCKED, AgentStatus.FAILED, AgentStatus.TERMINATED}),
    AgentStatus.FAILED: frozenset({AgentStatus.TERMINATED}),
    AgentStatus.COMPLETED: frozenset({AgentStatus.TERMINATED}),
    AgentStatus.TERMINATED: frozenset(),
}

APPROVAL_TRANSITIONS: Mapping[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.PENDING: frozenset({ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.CANCELLED}),
    ApprovalStatus.APPROVED: frozenset(),
    ApprovalStatus.REJECTED: frozenset(),
    ApprovalStatus.CANCELLED: frozenset(),
}

TRANSITION_RULES: Mapping[type[Enum], Mapping[Enum, frozenset[Enum]]] = {
    MissionStatus: MISSION_TRANSITIONS,
    TaskStatus: TASK_TRANSITIONS,
    AgentStatus: AGENT_TRANSITIONS,
    ApprovalStatus: APPROVAL_TRANSITIONS,
}


def can_transition(current: StatusType, target: StatusType) -> bool:
    """Return whether a typed status may transition to another typed status."""
    if type(current) is not type(target):
        return False
    rules = TRANSITION_RULES.get(type(current))
    return rules is not None and target in rules[current]


def validate_transition(current: StatusType, target: StatusType) -> None:
    """Raise a domain error when a requested transition is not allowed."""
    if not can_transition(current, target):
        raise InvalidStateTransition(f"Cannot transition {current!s} to {target!s}")
