"""Optional NVIDIA telemetry collected through machine-readable nvidia-smi CSV."""

from __future__ import annotations

import csv
from io import StringIO
import subprocess

from ttea.logging import get_logger

from .models import GpuMetrics, GpuProfile

LOGGER = get_logger("telemetry.nvidia")
QUERY_FIELDS = "index,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw"
MIB = 1024 * 1024


def _number(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


class NvidiaTelemetryCollector:
    """Best-effort NVIDIA collector; expected local absence returns no GPUs."""

    def __init__(self, executable: str = "nvidia-smi", timeout_seconds: float = 3.0) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def collect(self) -> tuple[GpuMetrics, ...]:
        """Return parsed GPU metrics, or an empty tuple when NVIDIA is unavailable."""
        try:
            result = subprocess.run(
                [self.executable, f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            LOGGER.info("NVIDIA telemetry unavailable: nvidia-smi not found")
            return ()
        except subprocess.TimeoutExpired:
            LOGGER.warning("NVIDIA telemetry unavailable: nvidia-smi timed out")
            return ()
        if result.returncode != 0:
            LOGGER.info("NVIDIA telemetry unavailable: nvidia-smi returned %s", result.returncode)
            return ()
        return self._parse(result.stdout)

    def profiles(self) -> tuple[GpuProfile, ...]:
        """Derive static GPU profile fields from a current driver observation."""
        return tuple(
            GpuProfile(metric.index, metric.name, metric.total_vram_bytes, metric.source)
            for metric in self.collect()
        )

    @staticmethod
    def _parse(output: str) -> tuple[GpuMetrics, ...]:
        metrics: list[GpuMetrics] = []
        for row in csv.reader(StringIO(output)):
            if not row or len(row) != 8:
                if row:
                    LOGGER.warning("Ignoring malformed NVIDIA telemetry row")
                continue
            try:
                index = int(row[0].strip())
            except ValueError:
                LOGGER.warning("Ignoring NVIDIA telemetry row with invalid index")
                continue
            name = row[1].strip()
            if not name:
                LOGGER.warning("Ignoring NVIDIA telemetry row without a name")
                continue
            values = [_number(value) for value in row[2:]]
            try:
                metrics.append(
                    GpuMetrics(
                        index=index,
                        name=name,
                        utilization_percent=values[0],
                        total_vram_bytes=int(values[1] * MIB) if values[1] is not None else None,
                        used_vram_bytes=int(values[2] * MIB) if values[2] is not None else None,
                        free_vram_bytes=int(values[3] * MIB) if values[3] is not None else None,
                        temperature_c=values[4],
                        power_draw_watts=values[5],
                    )
                )
            except ValueError:
                LOGGER.warning("Ignoring NVIDIA telemetry row with invalid metric range")
        return tuple(metrics)
