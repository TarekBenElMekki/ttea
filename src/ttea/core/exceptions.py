"""Exceptions raised by deterministic T/TEA core rules."""


class CoreError(Exception):
    """Base exception for T/TEA core rule violations."""


class InvalidStateTransition(CoreError):
    """Raised when a requested lifecycle transition is not permitted."""


class DependencyError(CoreError):
    """Raised when task dependency references are invalid."""


class DependencyCycleError(DependencyError):
    """Raised when task dependencies contain a cycle."""
