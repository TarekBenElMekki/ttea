"""Read-only recovery service built on the SQLite repository facade."""

from __future__ import annotations

from ttea.core import (
    AgentStatus,
    ApprovalStatus,
    Mission,
    MissionStatus,
    TaskStatus,
    get_ready_tasks,
    validate_dependencies,
)
from ttea.core.exceptions import DependencyError
from ttea.logging import get_logger
from ttea.persistence import Repository

from .exceptions import RecoveryValidationError
from .models import RecoverySnapshot

LOGGER = get_logger("recovery.service")

TERMINAL_MISSION_STATUSES = frozenset({
    MissionStatus.COMPLETED,
    MissionStatus.FAILED,
    MissionStatus.CANCELLED,
})
INTERRUPTED_TASK_STATUSES = frozenset({
    TaskStatus.ASSIGNED,
    TaskStatus.RUNNING,
    TaskStatus.VERIFYING,
})
STALE_RUNTIME_AGENT_STATUSES = frozenset({
    AgentStatus.CREATED,
    AgentStatus.IDLE,
    AgentStatus.WORKING,
    AgentStatus.BLOCKED,
    AgentStatus.PAUSED,
})


class RecoveryService:
    """Reconstruct and validate persisted mission state without changing it."""

    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def recover_mission(self, mission_id: str) -> RecoverySnapshot | None:
        """Return a validated, read-only recovery snapshot or ``None`` if absent."""
        mission = self.repository.get_mission(mission_id)
        if mission is None:
            return None
        LOGGER.info("Recovering mission %s", mission_id)
        tasks = tuple(self.repository.list_tasks(mission_id))
        agents = tuple(self.repository.list_agents(mission_id))
        approvals = tuple(self.repository.list_approvals(mission_id))
        events = tuple(self.repository.list_events(mission_id))
        self._validate(mission, tasks, agents, approvals, events)
        snapshot = RecoverySnapshot(
            mission=mission,
            tasks=tasks,
            agents=agents,
            approvals=approvals,
            events=events,
            interrupted_task_ids=tuple(task.id for task in tasks if task.status in INTERRUPTED_TASK_STATUSES),
            stale_runtime_agent_ids=tuple(agent.id for agent in agents if agent.status in STALE_RUNTIME_AGENT_STATUSES),
            pending_approval_ids=tuple(approval.id for approval in approvals if approval.status is ApprovalStatus.PENDING),
            ready_task_ids=tuple(task.id for task in get_ready_tasks(tasks)),
        )
        LOGGER.info("Recovered mission %s", mission_id)
        return snapshot

    def list_recoverable_missions(self) -> list[Mission]:
        """List non-terminal missions in repository ordering."""
        return [
            mission
            for mission in self.repository.list_missions()
            if mission.status not in TERMINAL_MISSION_STATUSES
        ]

    @staticmethod
    def _validate(mission, tasks, agents, approvals, events) -> None:
        mission_id = mission.id
        for entity_name, entities in (
            ("task", tasks),
            ("agent", agents),
            ("approval", approvals),
            ("event", events),
        ):
            for entity in entities:
                if entity.mission_id != mission_id:
                    raise RecoveryValidationError(
                        f"Recovered {entity_name} {entity.id} does not belong to mission {mission_id}"
                    )
        try:
            validate_dependencies(tasks)
        except DependencyError as error:
            raise RecoveryValidationError(f"Invalid task dependencies for mission {mission_id}: {error}") from error
        agent_ids = {agent.id for agent in agents}
        for task in tasks:
            if task.assigned_agent_id is not None and task.assigned_agent_id not in agent_ids:
                raise RecoveryValidationError(
                    f"Task {task.id} references missing mission agent {task.assigned_agent_id}"
                )
