"""Suite registry and failure-isolating validation runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import platform
import sys
from typing import Callable
from uuid import uuid4

from .models import ValidationRunResult, ValidationScenarioResult, ValidationStatus, freeze_evidence, overall_status
from .reporting import write_reports
from .scenario import ScenarioInterrupted, ValidationScenario


@dataclass(frozen=True)
class ValidationSuite:
    name: str
    scenario_factories: tuple[Callable[[], ValidationScenario], ...]
    context_factory: Callable[[], object]
    guard_factory: Callable[[object], object] | None = None


@dataclass(frozen=True)
class ValidationRunOptions:
    allow_model_download: bool = False
    pre_model_passed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_model_download, bool) or not isinstance(self.pre_model_passed, bool):
            raise ValueError("validation run options must be bools")


class SuiteRegistry:
    def __init__(self) -> None:
        self._suites: dict[str, ValidationSuite] = {}

    def register(self, suite: ValidationSuite) -> None:
        if not isinstance(suite.name, str) or not suite.name.strip():
            raise ValueError("suite name must be non-empty")
        if suite.name in self._suites:
            raise ValueError(f"validation suite {suite.name!r} is already registered")
        self._suites[suite.name] = suite

    def get(self, name: str) -> ValidationSuite:
        try:
            return self._suites[name]
        except KeyError as error:
            raise ValueError(f"unknown validation suite: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._suites))


class ValidationRunner:
    def __init__(self, registry: SuiteRegistry, report_root: Path = Path("data/validation")) -> None:
        self.registry, self.report_root = registry, report_root

    def run(self, suite_name: str, wait_for_resources_seconds: float = 0.0,
            options: ValidationRunOptions | None = None) -> ValidationRunResult:
        if not isinstance(wait_for_resources_seconds, (int, float)) or isinstance(wait_for_resources_seconds, bool) or not math.isfinite(wait_for_resources_seconds) or wait_for_resources_seconds < 0:
            raise ValueError("wait_for_resources_seconds must be a finite non-negative number")
        suite = self.registry.get(suite_name)
        options = options or ValidationRunOptions()
        if not isinstance(options, ValidationRunOptions):
            raise ValueError("options must be ValidationRunOptions")
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
        started = datetime.now(timezone.utc)
        context = suite.context_factory()
        configure = getattr(context, "configure_validation", None)
        if configure is not None:
            configure(options)
        guard = suite.guard_factory(context) if suite.guard_factory is not None else None
        results: list[ValidationScenarioResult] = []
        cleanup_verified, interrupted = True, False
        environment = {"python": sys.version.split()[0], "platform": platform.platform(), "executable": sys.executable}
        guard_established = False
        try:
            if guard is not None:
                try:
                    guard.establish()
                    guard_established = True
                    environment.update(getattr(guard, "environment_summary", {}))
                except KeyboardInterrupt:
                    interrupted = True
                    now = datetime.now(timezone.utc)
                    results.append(ValidationScenarioResult("global_run_guard", ValidationStatus.FAIL,
                                                          ("validation interrupted during global precheck",), now, now))
                except Exception as error_value:
                    now = datetime.now(timezone.utc)
                    results.append(ValidationScenarioResult("global_run_guard", ValidationStatus.BLOCKED,
                                                          (str(error_value),), now, now))
            if not interrupted and (guard is None or guard_established):
                for factory in suite.scenario_factories:
                    try:
                        results.append(factory().run(context, wait_for_resources_seconds))
                    except ScenarioInterrupted as interrupted_error:
                        results.append(interrupted_error.result)
                        interrupted = True
                        break
        finally:
            if guard is not None and guard_established:
                try:
                    guard.finalize()
                except BaseException as error_value:
                    cleanup_verified = False
                    now = datetime.now(timezone.utc)
                    results.append(ValidationScenarioResult("global_cleanup", ValidationStatus.FAIL,
                                                          (f"cleanup verification failed: {error_value}",), now, now))
        finished = datetime.now(timezone.utc)
        result_tuple = tuple(results)
        status = overall_status(result_tuple, cleanup_verified)
        result = ValidationRunResult(run_id, suite_name, started, finished, status, result_tuple,
                                     freeze_evidence(environment), cleanup_verified, interrupted=interrupted)
        return write_reports(result, self.report_root)


def exit_code(result: ValidationRunResult) -> int:
    if result.interrupted:
        return 130
    if result.overall_status is ValidationStatus.PASS:
        return 0
    if result.overall_status is ValidationStatus.BLOCKED:
        return 2
    return 1
