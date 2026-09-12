"""Exceptions for recovery analysis failures."""


class RecoveryError(Exception):
    """Base exception for recovery operations."""


class RecoveryValidationError(RecoveryError):
    """Raised when persisted mission state is structurally inconsistent."""
