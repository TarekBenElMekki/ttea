"""Provider-neutral contracts. Execution is deliberately disabled."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ttea.resources import ResourceRequest

from .models import TournamentCandidate, TournamentRound, TournamentTask


@dataclass(frozen=True)
class ProviderAvailability:
    available: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool) or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("invalid provider availability")


@dataclass(frozen=True)
class ProviderMetadata:
    provider: str
    model: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip() or not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("provider and model must be non-empty")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True)
class ProvisioningEligibility:
    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool) or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("invalid provisioning eligibility")


@dataclass(frozen=True)
class TournamentExecutionRequest:
    candidate_id: str
    round_id: str
    task_id: str
    repetition: int
    provider: str
    model: str
    prompt: str
    context_limit: int
    output_limit: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        for name in ("candidate_id", "round_id", "task_id", "provider", "model", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("repetition", "context_limit", "output_limit"):
            value = getattr(self, name)
            minimum = 0 if name == "repetition" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"invalid {name}")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class TournamentProvider(ABC):
    @abstractmethod
    def availability(self, candidate: TournamentCandidate) -> ProviderAvailability:
        raise NotImplementedError

    @abstractmethod
    def metadata(self, candidate: TournamentCandidate) -> ProviderMetadata:
        raise NotImplementedError

    @abstractmethod
    def provisioning_eligibility(self, candidate: TournamentCandidate) -> ProvisioningEligibility:
        raise NotImplementedError

    @abstractmethod
    def resource_request(self, candidate: TournamentCandidate, tournament_round: TournamentRound, task: TournamentTask) -> ResourceRequest:
        raise NotImplementedError

    @abstractmethod
    def build_execution_request(self, candidate: TournamentCandidate, tournament_round: TournamentRound, task: TournamentTask, repetition: int) -> TournamentExecutionRequest:
        raise NotImplementedError


class TournamentExecutionDisabled(RuntimeError):
    pass


def execute(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise TournamentExecutionDisabled("execution not enabled in framework milestone")
