"""Reusable validation scenario lifecycle with guaranteed cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any, Mapping

from .models import ValidationScenarioResult, ValidationStatus, freeze_evidence


@dataclass(frozen=True)
class ScenarioCheck:
    status: ValidationStatus | None = None
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    retryable_resource_delay: bool = False


@dataclass(frozen=True)
class ScenarioVerification:
    status: ValidationStatus = ValidationStatus.PASS
    reasons: tuple[str, ...] = ("scenario invariants verified",)
    evidence: Mapping[str, Any] = field(default_factory=dict)


class ValidationScenario:
    """Override lifecycle methods; cleanup is always invoked exactly once."""

    scenario_id = "scenario"

    def precheck(self, context) -> ScenarioCheck:
        return ScenarioCheck()

    def execute(self, context) -> Mapping[str, Any] | None:
        return None

    def verify(self, context, execution: Mapping[str, Any]) -> ScenarioVerification:
        return ScenarioVerification()

    def cleanup(self, context) -> None:
        return None

    def run(self, context, wait_for_resources_seconds: float = 0.0, retry_interval_seconds: float = 1.0) -> ValidationScenarioResult:
        if not isinstance(wait_for_resources_seconds, (int, float)) or isinstance(wait_for_resources_seconds, bool) or not math.isfinite(wait_for_resources_seconds) or wait_for_resources_seconds < 0:
            raise ValueError("wait_for_resources_seconds must be a finite non-negative number")
        started = datetime.now(timezone.utc)
        evidence: dict[str, Any] = {}
        status, reasons = ValidationStatus.FAIL, ("scenario did not complete",)
        interrupted = False
        try:
            deadline = time.monotonic() + wait_for_resources_seconds
            while True:
                check = self.precheck(context)
                evidence.update(check.evidence)
                if check.status is not ValidationStatus.BLOCKED or not check.retryable_resource_delay or time.monotonic() >= deadline:
                    break
                time.sleep(min(retry_interval_seconds, max(0.0, deadline - time.monotonic())))
            if check.status is not None:
                status = check.status
                reasons = check.reasons or (f"scenario {check.status.value.lower()}",)
            else:
                execution = dict(self.execute(context) or {})
                evidence.update(execution)
                verification = self.verify(context, execution)
                status, reasons = verification.status, verification.reasons
                evidence.update(verification.evidence)
        except KeyboardInterrupt:
            interrupted = True
            status, reasons = ValidationStatus.FAIL, ("validation interrupted by operator",)
        except BaseException as error_value:
            status = ValidationStatus.FAIL
            reasons = (f"unexpected {type(error_value).__name__}: {error_value}",)
        finally:
            try:
                self.cleanup(context)
            except BaseException as error_value:
                status = ValidationStatus.FAIL
                reasons = reasons + (f"cleanup failed: {type(error_value).__name__}: {error_value}",)
        finished = datetime.now(timezone.utc)
        result = ValidationScenarioResult(self.scenario_id, status, reasons, started, finished, freeze_evidence(evidence))
        if interrupted:
            raise ScenarioInterrupted(result)
        return result


class ScenarioInterrupted(KeyboardInterrupt):
    def __init__(self, result: ValidationScenarioResult) -> None:
        self.result = result
        super().__init__("validation interrupted")
