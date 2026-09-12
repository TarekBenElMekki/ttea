"""Policy and approval-workflow errors."""


class PolicyError(Exception):
    """Base exception for deterministic policy operations."""


class ApprovalWorkflowError(PolicyError):
    """Raised when an approval workflow operation is not permitted."""
