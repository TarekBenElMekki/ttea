from datetime import datetime, timezone
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

from ttea.telemetry import HardwareProfile, HardwareSnapshot, HardwareTelemetryService
from ttea.telemetry.models import CpuMetrics, DiskMetrics, GpuMetrics, GpuProfile, MemoryMetrics
from ttea.telemetry.nvidia import MIB, NvidiaTelemetryCollector
from ttea.telemetry.system import CPU_SAMPLE_INTERVAL_SECONDS, cpu_temperature


class FakePsutil:
    cpu_intervals = []

    def __init__(self, temperatures=None):
        self.temperatures = temperatures or {}

    @staticmethod
    def virtual_memory():
        return SimpleNamespace(total=16_000, used=6_000, available=10_000, percent=37.5)

    @staticmethod
    def disk_usage(path):
        return SimpleNamespace(total=100_000, used=25_000, free=75_000, percent=25.0)

    @staticmethod
    def disk_io_counters():
        return SimpleNamespace(read_bytes=123, write_bytes=456)

    @staticmethod
    def cpu_count(logical):
        return 8 if logical else 4

    @staticmethod
    def cpu_percent(interval=None):
        FakePsutil.cpu_intervals.append(interval)
        return 42.0

    @staticmethod
    def cpu_freq():
        return SimpleNamespace(current=2500.0)

    @staticmethod
    def boot_time():
        return 1.0

    def sensors_temperatures(self):
        return self.temperatures


class FakeNvidiaCollector:
    def __init__(self, metrics=(), profiles=()):
        self._metrics = metrics
        self._profiles = profiles

    def collect(self):
        return self._metrics

    def profiles(self):
        return self._profiles


class TelemetryServiceTests(unittest.TestCase):
    def setUp(self):
        FakePsutil.cpu_intervals = []

    def test_profile_and_snapshot_use_measured_mocked_values(self):
        gpu_profile = GpuProfile(index=0, name="Mock GPU", total_vram_bytes=2 * MIB, source="nvidia-smi")
        service = HardwareTelemetryService(
            psutil_module=FakePsutil(),
            nvidia_collector=FakeNvidiaCollector(profiles=(gpu_profile,)),
        )

        profile = service.get_profile()
        snapshot = service.snapshot()

        self.assertIsInstance(profile, HardwareProfile)
        self.assertEqual(profile.physical_core_count, 4)
        self.assertEqual(profile.logical_core_count, 8)
        self.assertEqual(profile.memory_total_bytes, 16_000)
        self.assertEqual(profile.storage_total_bytes, 100_000)
        self.assertEqual(profile.gpus, (gpu_profile,))
        self.assertIsInstance(snapshot, HardwareSnapshot)
        self.assertEqual(snapshot.timestamp.tzinfo, timezone.utc)
        self.assertEqual(snapshot.cpu.utilization_percent, 42.0)
        self.assertEqual(FakePsutil.cpu_intervals, [CPU_SAMPLE_INTERVAL_SECONDS])
        self.assertEqual(snapshot.cpu.frequency_mhz, 2500.0)
        self.assertIsNone(snapshot.cpu.temperature_c)
        self.assertEqual(snapshot.cpu.temperature_source, "unavailable")
        self.assertEqual(snapshot.memory.available_bytes, 10_000)
        self.assertEqual(snapshot.disk.read_bytes, 123)
        self.assertEqual(snapshot.disk.write_bytes, 456)

    def test_credible_cpu_temperature_is_optional(self):
        temperatures = {"coretemp": [SimpleNamespace(current=61.5)]}
        snapshot = HardwareTelemetryService(psutil_module=FakePsutil(temperatures)).snapshot()

        self.assertEqual(snapshot.cpu.temperature_c, 61.5)
        self.assertEqual(snapshot.cpu.temperature_source, "psutil:coretemp")

    def test_cpu_temperature_provider_failures_degrade_safely(self):
        class MissingSensors:
            pass

        class NotImplementedSensors(FakePsutil):
            def sensors_temperatures(self):
                raise NotImplementedError

        class OSErrorSensors(FakePsutil):
            def sensors_temperatures(self):
                raise OSError("sensor unavailable")

        self.assertEqual(cpu_temperature(MissingSensors()), (None, "unavailable"))
        self.assertEqual(cpu_temperature(NotImplementedSensors()), (None, "unavailable"))
        self.assertEqual(cpu_temperature(OSErrorSensors()), (None, "unavailable"))

    def test_machine_without_nvidia_returns_valid_snapshot(self):
        snapshot = HardwareTelemetryService(
            psutil_module=FakePsutil(), nvidia_collector=FakeNvidiaCollector()
        ).snapshot()

        self.assertEqual(snapshot.gpus, ())
        self.assertIsNotNone(snapshot.cpu.utilization_percent)


class NvidiaCollectorTests(unittest.TestCase):
    def test_single_and_multiple_gpu_csv_parsing(self):
        output = "0, GeForce MX350, 25, 2048, 512, 1536, 62, 18.5\n1, RTX, 50, 4096, 1024, 3072, 70, 45"
        metrics = NvidiaTelemetryCollector._parse(output)

        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].name, "GeForce MX350")
        self.assertEqual(metrics[0].utilization_percent, 25.0)
        self.assertEqual(metrics[0].total_vram_bytes, 2048 * MIB)
        self.assertEqual(metrics[0].used_vram_bytes, 512 * MIB)
        self.assertEqual(metrics[0].free_vram_bytes, 1536 * MIB)
        self.assertEqual(metrics[0].temperature_c, 62.0)
        self.assertEqual(metrics[0].power_draw_watts, 18.5)
        self.assertEqual(metrics[1].index, 1)

    def test_na_values_remain_unavailable_not_zero(self):
        metric = NvidiaTelemetryCollector._parse("0, GPU, [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]")[0]

        self.assertIsNone(metric.utilization_percent)
        self.assertIsNone(metric.total_vram_bytes)
        self.assertIsNone(metric.temperature_c)
        self.assertIsNone(metric.power_draw_watts)

    def test_negative_temperature_is_not_rejected_as_impossible(self):
        metric = GpuMetrics(0, "GPU", 0, None, None, None, -5.0, None)
        self.assertEqual(metric.temperature_c, -5.0)

    def test_missing_timeout_and_malformed_nvidia_data_are_safe(self):
        collector = NvidiaTelemetryCollector()
        with patch("ttea.telemetry.nvidia.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(collector.collect(), ())
        with patch("ttea.telemetry.nvidia.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 3)):
            self.assertEqual(collector.collect(), ())
        with patch("ttea.telemetry.nvidia.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "not,csv", "")):
            self.assertEqual(collector.collect(), ())


class TelemetryModelTests(unittest.TestCase):
    def test_range_validation_rejects_invalid_percentages(self):
        with self.assertRaises(ValueError):
            CpuMetrics(utilization_percent=101, frequency_mhz=None, temperature_c=None)
        with self.assertRaises(ValueError):
            MemoryMetrics(total_bytes=1, used_bytes=0, available_bytes=1, utilization_percent=-1)
        with self.assertRaises(ValueError):
            DiskMetrics(total_bytes=1, used_bytes=0, free_bytes=1, utilization_percent=0, read_bytes=-1, write_bytes=None)
        with self.assertRaises(ValueError):
            HardwareSnapshot(
                timestamp=datetime.now(),
                cpu=CpuMetrics(None, None, None),
                memory=MemoryMetrics(1, 0, 1, 0),
                disk=DiskMetrics(1, 0, 1, 0, None, None),
                gpus=(),
                uptime_seconds=None,
            )
