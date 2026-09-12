"""Persistence-specific errors."""


class PersistenceError(Exception):
    """Base exception for persistence operations with domain context."""


class DuplicateEventError(PersistenceError):
    """Raised when attempting to append an event ID that already exists."""
