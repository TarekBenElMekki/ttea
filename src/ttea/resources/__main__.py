"""Print the current safe resource budget."""

from .manager import ResourceManager
from ttea.telemetry import HardwareTelemetryService


def _value(value):
    return "unavailable" if value is None else f"{value} bytes" if isinstance(value, int) else str(value)


def main() -> None:
    budget = ResourceManager(HardwareTelemetryService()).current_budget()
    print("T/TEA Resource Budget\n----------------------")
    print(f"RAM available: {budget.available_ram_bytes} bytes\nRAM reserved: {budget.reserved_ram_bytes} bytes\nRAM allocatable: {budget.allocatable_ram_bytes} bytes")
    print(f"\nDisk free: {budget.free_disk_bytes} bytes\nDisk reserved: {budget.reserved_disk_bytes} bytes\nDisk allocatable: {budget.allocatable_disk_bytes} bytes")
    for gpu in budget.gpus:
        print(f"\nGPU {gpu.index}: {gpu.name}\nVRAM free: {_value(gpu.free_vram_bytes)}\nVRAM reserved: {gpu.reserved_vram_bytes} bytes\nVRAM allocatable: {gpu.allocatable_vram_bytes} bytes\nTemperature: {_value(gpu.temperature_c)}\nTemperature threshold: {gpu.max_temperature_c} C")


if __name__ == "__main__":
    main()
