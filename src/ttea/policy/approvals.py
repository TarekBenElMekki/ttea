"""Approval request and resolution workflow primitives without persistence."""

from __future__ import annotations

from ttea.core import Approval, ApprovalStatus, Event, resolve_approval
from ttea.logging import get_logger

from .enums import ExecutionPolicyLevel
from .engine import ExecutionPolicy
from .exceptions import ApprovalWorkflowError
from .models import ActionRequest, PolicyDecision

LOGGER = get_logger("policy.approvals")


def request_approval(
    mission_id: str,
    requested_action: str,
    reason: str,
    approval_type: str = "execution",
) -> tuple[Approval, Event]:
    """Create a pending approval and matching request event without persistence."""
    approval = Approval(
        mission_id=mission_id,
        approval_type=approval_type,
        requested_action=requested_action,
        reason=reason,
    )
    event = Event(
        event_type="approval.requested",
        mission_id=mission_id,
        payload={
            "approval_id": approval.id,
            "approval_type": approval.approval_type,
            "requested_action": approval.requested_action,
        },
    )
    LOGGER.info("Approval %s requested for mission %s", approval.id, mission_id)
    return approval, event


def request_approval_for_decision(
    request: ActionRequest,
    decision: PolicyDecision,
    policy: ExecutionPolicy,
) -> tuple[Approval, Event]:
    """Create an approval only when a trusted policy confirms the decision."""
    if decision.action_request_id != request.id or decision.operation_type != request.operation_type:
        raise ApprovalWorkflowError("Policy decision does not correspond to the action request")
    trusted_decision = policy.evaluate(request)
    if decision != trusted_decision:
        raise ApprovalWorkflowError("Policy decision does not match trusted policy evaluation")
    if trusted_decision.level is not ExecutionPolicyLevel.APPROVAL:
        raise ApprovalWorkflowError(
            f"Cannot create approval for policy level {trusted_decision.level.value}"
        )
    target = f" targeting {request.target}" if request.target is not None else ""
    requested_action = f"{request.operation_type}: {request.description}{target}"
    return request_approval(request.mission_id, requested_action, trusted_decision.reason)


def approve_approval(approval: Approval, reason: str | None = None) -> Event:
    """Approve through the authoritative core state service."""
    return resolve_approval(approval, ApprovalStatus.APPROVED, reason)


def reject_approval(approval: Approval, reason: str | None = None) -> Event:
    """Reject through the authoritative core state service."""
    return resolve_approval(approval, ApprovalStatus.REJECTED, reason)


def cancel_approval(approval: Approval, reason: str | None = None) -> Event:
    """Cancel through the authoritative core state service."""
    return resolve_approval(approval, ApprovalStatus.CANCELLED, reason)
