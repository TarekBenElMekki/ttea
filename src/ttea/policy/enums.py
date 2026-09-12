"""Execution-policy classifications."""

from enum import Enum


class ExecutionPolicyLevel(str, Enum):
    AUTO = "auto"
    REVIEW = "review"
    APPROVAL = "approval"
    FORBIDDEN = "forbidden"
