"""Immutable, validated hardware profile and observation models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _nonnegative(value: int | float | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, (int, float)) or value < 0):
        raise ValueError(f"{field_name} must be non-negative when available")


def _percentage(value: float | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 100):
        raise ValueError(f"{field_name} must be between 0 and 100 when available")


def _numeric(value: float | None, field_name: str) -> None:
    if value is not None and not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric when available")


@dataclass(frozen=True)
class GpuProfile:
    index: int
    name: str
    total_vram_bytes: int | None
    source: str

    def __post_init__(self) -> None:
        _nonnegative(self.index, "index")
        _nonnegative(self.total_vram_bytes, "total_vram_bytes")


@dataclass(frozen=True)
class HardwareProfile:
    hostname: str
    operating_system: str
    os_release: str
    os_version: str
    machine_architecture: str
    cpu_model: str | None
    physical_core_count: int | None
    logical_core_count: int | None
    memory_total_bytes: int
    storage_total_bytes: int
    gpus: tuple[GpuProfile, ...]
    source: str = "psutil"

    def __post_init__(self) -> None:
        _nonnegative(self.physical_core_count, "physical_core_count")
        _nonnegative(self.logical_core_count, "logical_core_count")
        _nonnegative(self.memory_total_bytes, "memory_total_bytes")
        _nonnegative(self.storage_total_bytes, "storage_total_bytes")


@dataclass(frozen=True)
class CpuMetrics:
    utilization_percent: float | None
    frequency_mhz: float | None
    temperature_c: float | None
    source: str = "psutil"
    temperature_source: str = "unavailable"

    def __post_init__(self) -> None:
        _percentage(self.utilization_percent, "utilization_percent")
        _nonnegative(self.frequency_mhz, "frequency_mhz")
        _numeric(self.temperature_c, "temperature_c")


@dataclass(frozen=True)
class MemoryMetrics:
    total_bytes: int
    used_bytes: int
    available_bytes: int
    utilization_percent: float | None
    source: str = "psutil"

    def __post_init__(self) -> None:
        for field_name, value in (("total_bytes", self.total_bytes), ("used_bytes", self.used_bytes), ("available_bytes", self.available_bytes)):
            _nonnegative(value, field_name)
        _percentage(self.utilization_percent, "utilization_percent")


@dataclass(frozen=True)
class DiskMetrics:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    utilization_percent: float | None
    read_bytes: int | None
    write_bytes: int | None
    source: str = "psutil"

    def __post_init__(self) -> None:
        for field_name, value in (("total_bytes", self.total_bytes), ("used_bytes", self.used_bytes), ("free_bytes", self.free_bytes), ("read_bytes", self.read_bytes), ("write_bytes", self.write_bytes)):
            _nonnegative(value, field_name)
        _percentage(self.utilization_percent, "utilization_percent")


@dataclass(frozen=True)
class GpuMetrics:
    index: int
    name: str
    utilization_percent: float | None
    total_vram_bytes: int | None
    used_vram_bytes: int | None
    free_vram_bytes: int | None
    temperature_c: float | None
    power_draw_watts: float | None
    source: str = "nvidia-smi"

    def __post_init__(self) -> None:
        _nonnegative(self.index, "index")
        _percentage(self.utilization_percent, "utilization_percent")
        for field_name, value in (("total_vram_bytes", self.total_vram_bytes), ("used_vram_bytes", self.used_vram_bytes), ("free_vram_bytes", self.free_vram_bytes), ("power_draw_watts", self.power_draw_watts)):
            _nonnegative(value, field_name)
        _numeric(self.temperature_c, "temperature_c")


@dataclass(frozen=True)
class HardwareSnapshot:
    timestamp: datetime
    cpu: CpuMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    gpus: tuple[GpuMetrics, ...]
    uptime_seconds: float | None
    system_source: str = "psutil"

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timezone.utc.utcoffset(self.timestamp):
            raise ValueError("timestamp must be timezone-aware UTC")
        _nonnegative(self.uptime_seconds, "uptime_seconds")
