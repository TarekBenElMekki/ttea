"""Typed runtime primitives for the T/TEA core."""

from .enums import (
    AgentStatus,
    ApprovalStatus,
    EventSeverity,
    MissionStatus,
    TaskStatus,
)
from .models import AgentSpec, Approval, Event, Mission, Task
from .dependencies import dependencies_completed, get_ready_tasks, is_task_blocked, is_task_ready, validate_dependencies
from .exceptions import CoreError, DependencyCycleError, DependencyError, InvalidStateTransition
from .transitions import can_transition, validate_transition

__all__ = [
    "AgentSpec",
    "AgentStatus",
    "Approval",
    "ApprovalStatus",
    "can_transition",
    "CoreError",
    "dependencies_completed",
    "DependencyCycleError",
    "DependencyError",
    "Event",
    "EventSeverity",
    "get_ready_tasks",
    "InvalidStateTransition",
    "is_task_blocked",
    "is_task_ready",
    "Mission",
    "MissionStatus",
    "Task",
    "TaskStatus",
    "validate_dependencies",
    "validate_transition",
]
