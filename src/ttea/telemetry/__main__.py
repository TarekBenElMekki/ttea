"""Print a concise local-only T/TEA hardware diagnostic."""

from .service import HardwareTelemetryService


def _display(value) -> str:
    return "unavailable" if value is None else str(value)


def main() -> None:
    service = HardwareTelemetryService()
    profile = service.get_profile()
    snapshot = service.snapshot()
    print("T/TEA Hardware Profile")
    print("-----------------------")
    print(f"CPU: {_display(profile.cpu_model)}")
    print(f"Physical cores: {_display(profile.physical_core_count)}")
    print(f"Logical cores: {_display(profile.logical_core_count)}")
    print(f"RAM total: {profile.memory_total_bytes} bytes")
    for gpu in profile.gpus:
        print(f"GPU {gpu.index}: {gpu.name}")
        print(f"VRAM total: {_display(gpu.total_vram_bytes)} bytes")
    print("\nCurrent Snapshot")
    print("----------------")
    print(f"CPU utilization: {_display(snapshot.cpu.utilization_percent)}%")
    print(f"CPU frequency: {_display(snapshot.cpu.frequency_mhz)} MHz")
    print(f"CPU temperature: {_display(snapshot.cpu.temperature_c)}")
    print(f"RAM total: {snapshot.memory.total_bytes} bytes")
    print(f"RAM used: {snapshot.memory.used_bytes} bytes")
    print(f"RAM available: {snapshot.memory.available_bytes} bytes")
    print(f"RAM utilization: {_display(snapshot.memory.utilization_percent)}%")
    print(f"Disk total: {snapshot.disk.total_bytes} bytes")
    print(f"Disk used: {snapshot.disk.used_bytes} bytes")
    print(f"Disk free: {snapshot.disk.free_bytes} bytes")
    print(f"Disk utilization: {_display(snapshot.disk.utilization_percent)}%")
    for gpu in snapshot.gpus:
        print(f"GPU {gpu.index}: {gpu.name}")
        print(f"GPU {gpu.index} utilization: {_display(gpu.utilization_percent)}%")
        print(f"GPU {gpu.index} VRAM total: {_display(gpu.total_vram_bytes)} bytes")
        print(f"GPU {gpu.index} VRAM used: {_display(gpu.used_vram_bytes)} bytes")
        print(f"GPU {gpu.index} VRAM free: {_display(gpu.free_vram_bytes)} bytes")
        print(f"GPU {gpu.index} temperature: {_display(gpu.temperature_c)}")
        print(f"GPU {gpu.index} power draw: {_display(gpu.power_draw_watts)} watts")


if __name__ == "__main__":
    main()
