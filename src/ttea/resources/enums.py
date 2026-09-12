"""Resource-management decisions."""

from enum import Enum


class ResourceDecision(str, Enum):
    """The deterministic outcome of evaluating a resource request."""

    ALLOW = "ALLOW"
    DELAY = "DELAY"
    DENY = "DENY"
