"""Fresh-snapshot, deterministic resource evaluation."""

from __future__ import annotations

import math

from ttea.logging.setup import get_logger, setup_logging
from ttea.telemetry import HardwareProfile, HardwareSnapshot, HardwareTelemetryService

from .enums import ResourceDecision
from .models import GpuResourceBudget, ResourceAssessment, ResourceBudget, ResourcePolicy, ResourceRequest
from .policy import baseline_policy


def _budget(profile: HardwareProfile, snapshot: HardwareSnapshot, policy: ResourcePolicy) -> ResourceBudget:
    metrics_by_index = {gpu.index: gpu for gpu in snapshot.gpus}
    gpus = tuple(
        GpuResourceBudget(
            index=gpu.index, name=gpu.name, total_vram_bytes=gpu.total_vram_bytes,
            free_vram_bytes=metrics_by_index[gpu.index].free_vram_bytes if gpu.index in metrics_by_index else None,
            reserved_vram_bytes=policy.reserved_gpu_vram_bytes,
            allocatable_vram_bytes=max(0, (metrics_by_index[gpu.index].free_vram_bytes or 0) - policy.reserved_gpu_vram_bytes) if gpu.index in metrics_by_index and metrics_by_index[gpu.index].free_vram_bytes is not None else 0,
            temperature_c=_finite_temperature(metrics_by_index[gpu.index].temperature_c) if gpu.index in metrics_by_index else None,
            max_temperature_c=policy.max_gpu_temperature_c,
        ) for gpu in profile.gpus
    )
    return ResourceBudget(snapshot.timestamp, snapshot.memory.available_bytes, policy.reserved_ram_bytes,
                          max(0, snapshot.memory.available_bytes - policy.reserved_ram_bytes),
                          snapshot.disk.free_bytes, policy.reserved_disk_bytes,
                          max(0, snapshot.disk.free_bytes - policy.reserved_disk_bytes), gpus)


def _finite_temperature(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def assess_resources(profile: HardwareProfile, snapshot: HardwareSnapshot, request: ResourceRequest, policy: ResourcePolicy) -> ResourceAssessment:
    """Evaluate one trusted estimate without mutating source telemetry or policy."""
    deny, delay = [], []
    ram_headroom = snapshot.memory.available_bytes - policy.reserved_ram_bytes - request.estimated_ram_bytes
    disk_headroom = snapshot.disk.free_bytes - policy.reserved_disk_bytes - request.estimated_disk_bytes
    if request.estimated_ram_bytes + policy.reserved_ram_bytes > profile.memory_total_bytes:
        deny.append("RAM requirement plus reserve exceeds physical RAM capacity")
    elif ram_headroom < 0:
        delay.append("current available RAM cannot satisfy requirement plus reserve")
    if request.estimated_disk_bytes + policy.reserved_disk_bytes > profile.storage_total_bytes:
        deny.append("disk requirement plus reserve exceeds total disk capacity")
    elif disk_headroom < 0:
        delay.append("current free disk cannot satisfy requirement plus reserve")

    selected = None
    gpu_total = gpu_free = gpu_reserve = gpu_headroom = gpu_temperature = None
    if request.estimated_gpu_vram_bytes:
        profiles = {gpu.index: gpu for gpu in profile.gpus}
        if request.gpu_index is not None:
            selected = profiles.get(request.gpu_index)
            if selected is None:
                deny.append(f"requested GPU {request.gpu_index} does not exist in hardware profile")
        else:
            fitting = [gpu for gpu in profile.gpus if gpu.total_vram_bytes is not None and request.estimated_gpu_vram_bytes + policy.reserved_gpu_vram_bytes <= gpu.total_vram_bytes]
            unknown = [gpu for gpu in profile.gpus if gpu.total_vram_bytes is None]
            if len(fitting) == 1:
                selected = fitting[0]
            elif len(fitting) > 1:
                delay.append("multiple GPUs can satisfy the request; explicit GPU selection is required")
            elif unknown:
                delay.append("GPU total VRAM is unavailable, so GPU selection cannot be confirmed")
            else:
                deny.append("no GPU has enough total VRAM for requirement plus reserve")
        if selected is not None:
            gpu_total, gpu_reserve = selected.total_vram_bytes, policy.reserved_gpu_vram_bytes
            if gpu_total is None:
                delay.append(f"GPU {selected.index} total VRAM is unavailable")
            elif request.estimated_gpu_vram_bytes + gpu_reserve > gpu_total:
                deny.append(f"GPU {selected.index} VRAM requirement plus reserve exceeds total VRAM")
            metric = next((item for item in snapshot.gpus if item.index == selected.index), None)
            if metric is None:
                delay.append(f"current telemetry for GPU {selected.index} is unavailable")
            else:
                gpu_free, gpu_temperature = metric.free_vram_bytes, _finite_temperature(metric.temperature_c)
                gpu_headroom = None if gpu_free is None else gpu_free - gpu_reserve - request.estimated_gpu_vram_bytes
                if gpu_free is None:
                    delay.append(f"current free VRAM for GPU {selected.index} is unavailable")
                elif gpu_headroom < 0:
                    delay.append(f"current free VRAM on GPU {selected.index} cannot satisfy requirement plus reserve")
                if gpu_temperature is not None and gpu_temperature > policy.max_gpu_temperature_c:
                    delay.append(f"GPU {selected.index} temperature exceeds policy threshold")
                elif gpu_temperature is None and policy.require_gpu_temperature_for_gpu_workloads:
                    delay.append(f"GPU {selected.index} temperature is required but unavailable")
    decision = ResourceDecision.DENY if deny else ResourceDecision.DELAY if delay else ResourceDecision.ALLOW
    reasons = tuple(deny + delay) or ("all requested resources satisfy current policy",)
    return ResourceAssessment(request.id, decision, reasons, snapshot.timestamp, request.estimated_ram_bytes,
                              snapshot.memory.available_bytes, policy.reserved_ram_bytes, ram_headroom,
                              request.estimated_disk_bytes, snapshot.disk.free_bytes, policy.reserved_disk_bytes,
                              disk_headroom, selected.index if selected else None, gpu_total, gpu_free, gpu_reserve,
                              gpu_headroom, gpu_temperature)


class ResourceManager:
    """Coordinates fresh passive telemetry with deterministic policy decisions."""

    def __init__(self, telemetry_service: HardwareTelemetryService, policy: ResourcePolicy | None = None) -> None:
        self.telemetry_service = telemetry_service
        self.policy = policy
        setup_logging()
        self.logger = get_logger("resources")

    def _observe(self) -> tuple[HardwareProfile, HardwareSnapshot, ResourcePolicy]:
        profile = self.telemetry_service.get_profile()
        snapshot = self.telemetry_service.snapshot()
        return profile, snapshot, self.policy or baseline_policy(profile)

    def current_budget(self) -> ResourceBudget:
        profile, snapshot, policy = self._observe()
        budget = _budget(profile, snapshot, policy)
        self.logger.info("resource budget sampled timestamp=%s", snapshot.timestamp.isoformat())
        return budget

    def assess(self, request: ResourceRequest) -> ResourceAssessment:
        profile, snapshot, policy = self._observe()
        assessment = assess_resources(profile, snapshot, request, policy)
        self.logger.info("resource assessment request_id=%s decision=%s", request.id, assessment.decision.value)
        return assessment
