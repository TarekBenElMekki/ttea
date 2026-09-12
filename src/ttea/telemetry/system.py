"""Cross-platform system measurements built on psutil."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import socket
import time
from typing import Any

from .models import CpuMetrics, DiskMetrics, HardwareProfile, HardwareSnapshot, MemoryMetrics

CREDIBLE_CPU_SENSOR_NAMES = frozenset({"coretemp", "k10temp", "cpu_thermal"})
CPU_SAMPLE_INTERVAL_SECONDS = 0.1


def cpu_temperature(psutil_module: Any) -> tuple[float | None, str]:
    """Return a credible CPU temperature only when a supported provider exposes one."""
    sensors = getattr(psutil_module, "sensors_temperatures", None)
    if sensors is None:
        return None, "unavailable"
    try:
        readings = sensors() or {}
    except (NotImplementedError, OSError):
        return None, "unavailable"
    for sensor_name in CREDIBLE_CPU_SENSOR_NAMES:
        for reading in readings.get(sensor_name, []):
            current = getattr(reading, "current", None)
            if isinstance(current, (int, float)) and current >= 0:
                return float(current), f"psutil:{sensor_name}"
    return None, "unavailable"


def build_profile(psutil_module: Any, gpu_profiles: tuple, project_path: Path) -> HardwareProfile:
    """Build one static machine profile without dynamic utilization values."""
    memory = psutil_module.virtual_memory()
    disk = psutil_module.disk_usage(str(project_path))
    processor = platform.processor() or None
    return HardwareProfile(
        hostname=socket.gethostname(),
        operating_system=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine_architecture=platform.machine(),
        cpu_model=processor,
        physical_core_count=psutil_module.cpu_count(logical=False),
        logical_core_count=psutil_module.cpu_count(logical=True),
        memory_total_bytes=memory.total,
        storage_total_bytes=disk.total,
        gpus=gpu_profiles,
    )


def build_snapshot(psutil_module: Any, gpu_metrics: tuple, project_path: Path) -> HardwareSnapshot:
    """Sample current CPU, memory, disk, uptime, and optional GPU measurements."""
    memory = psutil_module.virtual_memory()
    disk = psutil_module.disk_usage(str(project_path))
    disk_io = psutil_module.disk_io_counters()
    frequency = psutil_module.cpu_freq()
    temperature, temperature_source = cpu_temperature(psutil_module)
    return HardwareSnapshot(
        timestamp=datetime.now(timezone.utc),
        cpu=CpuMetrics(
            utilization_percent=psutil_module.cpu_percent(interval=CPU_SAMPLE_INTERVAL_SECONDS),
            frequency_mhz=getattr(frequency, "current", None) if frequency is not None else None,
            temperature_c=temperature,
            temperature_source=temperature_source,
        ),
        memory=MemoryMetrics(memory.total, memory.used, memory.available, memory.percent),
        disk=DiskMetrics(
            disk.total,
            disk.used,
            disk.free,
            disk.percent,
            getattr(disk_io, "read_bytes", None) if disk_io is not None else None,
            getattr(disk_io, "write_bytes", None) if disk_io is not None else None,
        ),
        gpus=gpu_metrics,
        uptime_seconds=max(0.0, time.time() - psutil_module.boot_time()),
    )
