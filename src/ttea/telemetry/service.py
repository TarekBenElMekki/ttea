"""Explicit, passive local hardware telemetry service."""

from __future__ import annotations

from pathlib import Path

import psutil

from .models import HardwareProfile, HardwareSnapshot
from .nvidia import NvidiaTelemetryCollector
from .system import build_profile, build_snapshot


class HardwareTelemetryService:
    """Collect profile and point-in-time measurements without polling or policy decisions."""

    def __init__(self, project_path: Path | None = None, psutil_module=psutil, nvidia_collector: NvidiaTelemetryCollector | None = None) -> None:
        self.project_path = project_path or Path.cwd()
        self.psutil_module = psutil_module
        self.nvidia_collector = nvidia_collector or NvidiaTelemetryCollector()

    def get_profile(self) -> HardwareProfile:
        """Explicitly collect a fresh static hardware profile."""
        return build_profile(self.psutil_module, self.nvidia_collector.profiles(), self.project_path)

    def snapshot(self) -> HardwareSnapshot:
        """Explicitly sample current local hardware telemetry."""
        return build_snapshot(self.psutil_module, self.nvidia_collector.collect(), self.project_path)
