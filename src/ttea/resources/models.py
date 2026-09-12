"""Immutable request, policy, budget, and assessment models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from uuid import uuid4

from .enums import ResourceDecision


MIB = 1024 * 1024
GIB = 1024 * MIB


def _new_id() -> str:
    return str(uuid4())


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _optional_nonnegative_integer(value: int | None, name: str) -> None:
    if value is not None:
        _nonnegative_integer(value, name)


def _finite_number(value: int | float, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or (positive and value <= 0):
        qualifier = "a finite positive number" if positive else "a finite number"
        raise ValueError(f"{name} must be {qualifier}")


def _utc_timestamp(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True)
class ResourceRequest:
    """Trusted runtime estimate for one domain-independent workload."""

    workload_type: str
    estimated_ram_bytes: int
    estimated_disk_bytes: int
    estimated_gpu_vram_bytes: int = 0
    gpu_index: int | None = None
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.workload_type, str) or not self.workload_type.strip():
            raise ValueError("workload_type must be a non-empty string")
        for name in ("estimated_ram_bytes", "estimated_disk_bytes", "estimated_gpu_vram_bytes"):
            _nonnegative_integer(getattr(self, name), name)
        if self.gpu_index is not None:
            _nonnegative_integer(self.gpu_index, "gpu_index")
            if self.estimated_gpu_vram_bytes == 0:
                raise ValueError("gpu_index requires estimated_gpu_vram_bytes greater than zero")


@dataclass(frozen=True)
class ResourcePolicy:
    """Capacity deliberately kept outside T/TEA allocation."""

    reserved_ram_bytes: int
    reserved_disk_bytes: int
    reserved_gpu_vram_bytes: int
    max_gpu_temperature_c: float
    require_gpu_temperature_for_gpu_workloads: bool = True

    def __post_init__(self) -> None:
        for name in ("reserved_ram_bytes", "reserved_disk_bytes", "reserved_gpu_vram_bytes"):
            _nonnegative_integer(getattr(self, name), name)
        _finite_number(self.max_gpu_temperature_c, "max_gpu_temperature_c", positive=True)
        if not isinstance(self.require_gpu_temperature_for_gpu_workloads, bool):
            raise ValueError("require_gpu_temperature_for_gpu_workloads must be a bool")


@dataclass(frozen=True)
class GpuResourceBudget:
    index: int
    name: str
    total_vram_bytes: int | None
    free_vram_bytes: int | None
    reserved_vram_bytes: int
    allocatable_vram_bytes: int
    temperature_c: float | None
    max_temperature_c: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.index, "index")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        for name in ("total_vram_bytes", "free_vram_bytes"):
            _optional_nonnegative_integer(getattr(self, name), name)
        for name in ("reserved_vram_bytes", "allocatable_vram_bytes"):
            _nonnegative_integer(getattr(self, name), name)
        if self.temperature_c is not None:
            _finite_number(self.temperature_c, "temperature_c")
        _finite_number(self.max_temperature_c, "max_temperature_c", positive=True)


@dataclass(frozen=True)
class ResourceBudget:
    snapshot_timestamp: datetime
    available_ram_bytes: int
    reserved_ram_bytes: int
    allocatable_ram_bytes: int
    free_disk_bytes: int
    reserved_disk_bytes: int
    allocatable_disk_bytes: int
    gpus: tuple[GpuResourceBudget, ...]

    def __post_init__(self) -> None:
        _utc_timestamp(self.snapshot_timestamp, "snapshot_timestamp")
        for name in ("available_ram_bytes", "reserved_ram_bytes", "allocatable_ram_bytes", "free_disk_bytes", "reserved_disk_bytes", "allocatable_disk_bytes"):
            _nonnegative_integer(getattr(self, name), name)


@dataclass(frozen=True)
class ResourceAssessment:
    request_id: str
    decision: ResourceDecision
    reasons: tuple[str, ...]
    snapshot_timestamp: datetime
    ram_required_bytes: int
    ram_available_bytes: int
    ram_reserved_bytes: int
    ram_headroom_bytes: int
    disk_required_bytes: int
    disk_free_bytes: int
    disk_reserved_bytes: int
    disk_headroom_bytes: int
    selected_gpu_index: int | None = None
    gpu_total_vram_bytes: int | None = None
    gpu_free_vram_bytes: int | None = None
    gpu_reserved_vram_bytes: int | None = None
    gpu_headroom_bytes: int | None = None
    gpu_temperature_c: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.decision, ResourceDecision):
            raise ValueError("decision must be a ResourceDecision")
        if not isinstance(self.reasons, tuple) or not self.reasons or any(not isinstance(reason, str) or not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must be a non-empty tuple of non-empty strings")
        _utc_timestamp(self.snapshot_timestamp, "snapshot_timestamp")
        for name in ("ram_required_bytes", "ram_available_bytes", "ram_reserved_bytes", "disk_required_bytes", "disk_free_bytes", "disk_reserved_bytes"):
            _nonnegative_integer(getattr(self, name), name)
        if self.selected_gpu_index is not None:
            _nonnegative_integer(self.selected_gpu_index, "selected_gpu_index")
        for name in ("gpu_total_vram_bytes", "gpu_free_vram_bytes", "gpu_reserved_vram_bytes"):
            _optional_nonnegative_integer(getattr(self, name), name)
        if self.gpu_temperature_c is not None:
            _finite_number(self.gpu_temperature_c, "gpu_temperature_c")
