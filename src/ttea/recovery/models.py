"""Immutable view of one persisted mission's recovery state."""

from __future__ import annotations

from dataclasses import dataclass

from ttea.core import AgentSpec, Approval, Event, Mission, Task


@dataclass(frozen=True)
class RecoverySnapshot:
    """Read-only reconstructed state and deterministic recovery classifications."""

    mission: Mission
    tasks: tuple[Task, ...]
    agents: tuple[AgentSpec, ...]
    approvals: tuple[Approval, ...]
    events: tuple[Event, ...]
    interrupted_task_ids: tuple[str, ...]
    stale_runtime_agent_ids: tuple[str, ...]
    pending_approval_ids: tuple[str, ...]
    ready_task_ids: tuple[str, ...]

    @property
    def requires_reconciliation(self) -> bool:
        """Whether interrupted work or stale runtime instances need future handling."""
        return bool(self.interrupted_task_ids or self.stale_runtime_agent_ids)

    @property
    def awaiting_approval(self) -> bool:
        """Whether the recovered mission has unresolved pending approvals."""
        return bool(self.pending_approval_ids)
