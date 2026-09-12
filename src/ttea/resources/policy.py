"""Deterministic reserve-policy construction."""

from __future__ import annotations

from ttea.telemetry import HardwareProfile

from .models import GIB, MIB, ResourcePolicy


def baseline_policy(profile: HardwareProfile) -> ResourcePolicy:
    """Create the conservative baseline: RAM 20% (512 MiB..4 GiB), disk 10% (5..50 GiB), and VRAM 10% of the smallest known GPU (128 MiB..1 GiB; 256 MiB if unknown), with an 80 C operational GPU threshold."""
    ram_reserve = max(512 * MIB, min(4 * GIB, profile.memory_total_bytes // 5))
    disk_reserve = max(5 * GIB, min(50 * GIB, profile.storage_total_bytes // 10))
    known_vram = [gpu.total_vram_bytes for gpu in profile.gpus if gpu.total_vram_bytes is not None]
    vram_basis = min(known_vram) if known_vram else None
    vram_reserve = 256 * MIB if vram_basis is None else max(128 * MIB, min(1 * GIB, vram_basis // 10))
    return ResourcePolicy(ram_reserve, disk_reserve, vram_reserve, 80.0)
