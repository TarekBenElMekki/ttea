"""Exact-match deterministic execution-policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from ttea.logging import get_logger

from .enums import ExecutionPolicyLevel
from .models import ActionRequest, PolicyDecision, PolicyRule

LOGGER = get_logger("policy.engine")
FALLBACK_REASON = "No baseline rule matches this operation; explicit approval is required"


class ExecutionPolicy:
    """Evaluate immutable action requests against exact canonical operation rules."""

    def __init__(self, rules: Iterable[PolicyRule]) -> None:
        self._rules: dict[str, PolicyRule] = {}
        for rule in rules:
            if rule.operation_type in self._rules:
                raise ValueError(f"Duplicate policy rule for {rule.operation_type}")
            self._rules[rule.operation_type] = rule

    def evaluate(self, request: ActionRequest) -> PolicyDecision:
        """Classify a request without mutating it or granting caller-selected access."""
        rule = self._rules.get(request.operation_type)
        if rule is None:
            decision = PolicyDecision(
                action_request_id=request.id,
                operation_type=request.operation_type,
                level=ExecutionPolicyLevel.APPROVAL,
                reason=FALLBACK_REASON,
            )
        else:
            decision = PolicyDecision(
                action_request_id=request.id,
                operation_type=request.operation_type,
                level=rule.level,
                reason=rule.reason,
                matched_rule_id=rule.rule_id,
            )
        LOGGER.info("Policy classified operation %s as %s", request.operation_type, decision.level.value)
        return decision


def baseline_policy() -> ExecutionPolicy:
    """Return T/TEA's explicit conservative baseline execution policy."""
    definitions = (
        ("filesystem.read", ExecutionPolicyLevel.AUTO),
        ("filesystem.write", ExecutionPolicyLevel.REVIEW),
        ("process.execute", ExecutionPolicyLevel.REVIEW),
        ("network.read", ExecutionPolicyLevel.REVIEW),
        ("filesystem.delete", ExecutionPolicyLevel.APPROVAL),
        ("package.install", ExecutionPolicyLevel.APPROVAL),
        ("network.write", ExecutionPolicyLevel.APPROVAL),
        ("communication.send", ExecutionPolicyLevel.APPROVAL),
        ("credential.access", ExecutionPolicyLevel.APPROVAL),
        ("system.modify", ExecutionPolicyLevel.FORBIDDEN),
    )
    return ExecutionPolicy(
        PolicyRule(
            rule_id=f"baseline.{operation_type}",
            operation_type=operation_type,
            level=level,
            reason=f"Baseline policy classifies {operation_type} as {level.value}",
        )
        for operation_type, level in definitions
    )
