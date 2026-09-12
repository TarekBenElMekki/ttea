from datetime import datetime, timezone
import unittest

from ttea.resources import ResourceDecision, ResourceManager, ResourcePolicy, ResourceRequest, assess_resources, baseline_policy
from ttea.resources.models import GpuResourceBudget, ResourceAssessment
from ttea.resources.models import GIB, MIB
from ttea.telemetry.models import CpuMetrics, DiskMetrics, GpuMetrics, GpuProfile, HardwareProfile, HardwareSnapshot, MemoryMetrics


def profile(gpus=(), ram=16 * GIB, disk=100 * GIB):
    return HardwareProfile("host", "os", "r", "v", "x", None, None, None, ram, disk, tuple(gpus))


def snapshot(gpus=(), ram=12 * GIB, disk=80 * GIB):
    return HardwareSnapshot(datetime.now(timezone.utc), CpuMetrics(None, None, None), MemoryMetrics(16 * GIB, 4 * GIB, ram, None), DiskMetrics(100 * GIB, 20 * GIB, disk, None, None, None), tuple(gpus), None)


class ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.policy = ResourcePolicy(1 * GIB, 5 * GIB, 256 * MIB, 80)

    def request(self, **values):
        data = dict(workload_type="work", estimated_ram_bytes=1 * GIB, estimated_disk_bytes=1 * GIB)
        data.update(values)
        return ResourceRequest(**data)

    def test_request_and_policy_validation(self):
        with self.assertRaises(ValueError): self.request(estimated_ram_bytes=-1)
        with self.assertRaises(ValueError): self.request(workload_type=" ")
        with self.assertRaises(ValueError): self.request(gpu_index=0)
        with self.assertRaises(ValueError): ResourcePolicy(-1, 0, 0, 80)
        for threshold in (float("nan"), float("inf"), float("-inf"), 0, -1):
            with self.assertRaises(ValueError): ResourcePolicy(0, 0, 0, threshold)

    def test_baseline_derivation_and_budget_floor(self):
        base = baseline_policy(profile())
        self.assertEqual(base.reserved_ram_bytes, int(3.2 * GIB))
        self.assertEqual(base.reserved_disk_bytes, 10 * GIB)
        service = FakeTelemetry(profile(), snapshot(ram=0, disk=0))
        budget = ResourceManager(service, self.policy).current_budget()
        self.assertEqual((budget.allocatable_ram_bytes, budget.allocatable_disk_bytes), (0, 0))

    def test_allow_delay_and_deny_ram_disk(self):
        self.assertEqual(assess_resources(profile(), snapshot(), self.request(), self.policy).decision, ResourceDecision.ALLOW)
        self.assertEqual(assess_resources(profile(), snapshot(ram=1 * GIB), self.request(), self.policy).decision, ResourceDecision.DELAY)
        self.assertEqual(assess_resources(profile(ram=1 * GIB), snapshot(), self.request(), self.policy).decision, ResourceDecision.DENY)
        self.assertEqual(assess_resources(profile(), snapshot(disk=5 * GIB), self.request(), self.policy).decision, ResourceDecision.DELAY)
        self.assertEqual(assess_resources(profile(disk=5 * GIB), snapshot(), self.request(), self.policy).decision, ResourceDecision.DENY)

    def test_gpu_failures_and_success(self):
        req = self.request(estimated_gpu_vram_bytes=1 * GIB)
        self.assertEqual(assess_resources(profile(), snapshot(), req, self.policy).decision, ResourceDecision.DENY)
        gpu = GpuProfile(3, "gpu", 2 * GIB, "test")
        good = GpuMetrics(3, "gpu", None, 2 * GIB, None, 1500 * MIB, 60, None)
        self.assertEqual(assess_resources(profile((gpu,)), snapshot((good,)), req, self.policy).decision, ResourceDecision.ALLOW)
        self.assertEqual(assess_resources(profile((gpu,)), snapshot((GpuMetrics(3, "gpu", None, 2 * GIB, None, 500 * MIB, 60, None),)), req, self.policy).decision, ResourceDecision.DELAY)
        self.assertEqual(assess_resources(profile((gpu,)), snapshot(), req, self.policy).decision, ResourceDecision.DELAY)
        self.assertEqual(assess_resources(profile((gpu,)), snapshot((GpuMetrics(3, "gpu", None, 2 * GIB, None, 1500 * MIB, 90, None),)), req, self.policy).decision, ResourceDecision.DELAY)

    def test_gpu_total_missing_temperature_and_multiple_selection(self):
        req = self.request(estimated_gpu_vram_bytes=1 * GIB)
        small = GpuProfile(1, "small", 1 * GIB, "test")
        self.assertEqual(assess_resources(profile((small,)), snapshot(), req, self.policy).decision, ResourceDecision.DENY)
        gpu = GpuProfile(2, "unknown", None, "test")
        self.assertEqual(assess_resources(profile((gpu,)), snapshot(), req, self.policy).decision, ResourceDecision.DELAY)
        first, second = GpuProfile(0, "a", 2 * GIB, "t"), GpuProfile(4, "b", 2 * GIB, "t")
        both = snapshot((GpuMetrics(0, "a", None, 2 * GIB, None, 2 * GIB, 50, None), GpuMetrics(4, "b", None, 2 * GIB, None, 2 * GIB, 50, None)))
        self.assertEqual(assess_resources(profile((first, second)), both, req, self.policy).decision, ResourceDecision.DELAY)
        explicit = self.request(estimated_gpu_vram_bytes=1 * GIB, gpu_index=4)
        self.assertEqual(assess_resources(profile((first, second)), both, explicit, self.policy).selected_gpu_index, 4)
        missing_temp = snapshot((GpuMetrics(0, "a", None, 2 * GIB, None, 2 * GIB, None, None),))
        self.assertEqual(assess_resources(profile((first,)), missing_temp, req, self.policy).decision, ResourceDecision.DELAY)

    def test_cpu_temperature_and_fresh_manager_snapshots(self):
        req = self.request()
        source_profile, first, second = profile(), snapshot(), snapshot(ram=1 * GIB)
        service = FakeTelemetry(source_profile, first, second)
        manager = ResourceManager(service, self.policy)
        self.assertEqual(manager.assess(req).decision, ResourceDecision.ALLOW)
        self.assertEqual(manager.assess(req).decision, ResourceDecision.DELAY)
        self.assertEqual(service.snapshot_calls, 2)
        self.assertEqual(source_profile.memory_total_bytes, 16 * GIB)

    def test_non_finite_required_gpu_temperature_delays_and_unusual_finite_temperature_is_valid(self):
        gpu = GpuProfile(0, "gpu", 2 * GIB, "test")
        req = self.request(estimated_gpu_vram_bytes=1 * GIB)
        invalid = GpuMetrics(0, "gpu", None, 2 * GIB, None, 2 * GIB, float("nan"), None)
        assessment = assess_resources(profile((gpu,)), snapshot((invalid,)), req, self.policy)
        self.assertEqual(assessment.decision, ResourceDecision.DELAY)
        self.assertIsNone(assessment.gpu_temperature_c)
        budget = GpuResourceBudget(0, "gpu", 2 * GIB, 2 * GIB, 0, 2 * GIB, -12.5, 80)
        self.assertEqual(budget.temperature_c, -12.5)

    def test_gpu_budget_validation(self):
        with self.assertRaises(ValueError): GpuResourceBudget(-1, "gpu", None, None, 0, 0, None, 80)
        with self.assertRaises(ValueError): GpuResourceBudget(0, "", None, None, 0, 0, None, 80)
        with self.assertRaises(ValueError): GpuResourceBudget(0, "gpu", -1, None, 0, 0, None, 80)
        with self.assertRaises(ValueError): GpuResourceBudget(0, "gpu", None, None, 0, 0, float("nan"), 80)
        with self.assertRaises(ValueError): GpuResourceBudget(0, "gpu", None, None, 0, 0, None, 0)

    def test_assessment_validation_and_negative_headroom(self):
        current = assess_resources(profile(), snapshot(ram=1 * GIB), self.request(), self.policy)
        self.assertLess(current.ram_headroom_bytes, 0)
        kwargs = dict(request_id="request", decision=ResourceDecision.ALLOW, reasons=("ok",), snapshot_timestamp=datetime.now(timezone.utc), ram_required_bytes=0, ram_available_bytes=0, ram_reserved_bytes=0, ram_headroom_bytes=-1, disk_required_bytes=0, disk_free_bytes=0, disk_reserved_bytes=0, disk_headroom_bytes=-1)
        self.assertEqual(ResourceAssessment(**kwargs).ram_headroom_bytes, -1)
        with self.assertRaises(ValueError): ResourceAssessment(**(kwargs | {"decision": "ALLOW"}))
        with self.assertRaises(ValueError): ResourceAssessment(**(kwargs | {"reasons": ()}))
        with self.assertRaises(ValueError): ResourceAssessment(**(kwargs | {"snapshot_timestamp": datetime.now()}))

    def test_assessment_is_deterministic_and_preserves_sources(self):
        gpu = GpuProfile(0, "gpu", 2 * GIB, "test")
        source_profile = profile((gpu,))
        source_snapshot = snapshot((GpuMetrics(0, "gpu", None, 2 * GIB, None, 2 * GIB, 50, None),))
        source_request = self.request(estimated_gpu_vram_bytes=1 * GIB)
        source_policy = self.policy
        before = (source_profile, source_snapshot, source_request, source_policy)
        first = assess_resources(source_profile, source_snapshot, source_request, source_policy)
        second = assess_resources(source_profile, source_snapshot, source_request, source_policy)
        self.assertEqual(first, second)
        self.assertEqual((source_profile, source_snapshot, source_request, source_policy), before)


class FakeTelemetry:
    def __init__(self, hardware_profile, *snapshots):
        self.hardware_profile, self.snapshots, self.snapshot_calls = hardware_profile, list(snapshots), 0
    def get_profile(self): return self.hardware_profile
    def snapshot(self):
        item = self.snapshots[min(self.snapshot_calls, len(self.snapshots) - 1)]
        self.snapshot_calls += 1
        return item


if __name__ == "__main__":
    unittest.main()
