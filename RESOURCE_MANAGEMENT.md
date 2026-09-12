# Resource Management

Telemetry measures the machine. The Resource Manager separately and deterministically decides whether a trusted runtime workload estimate may start. An LLM never makes this safety decision.

Non-finite thermal readings are treated as unavailable and unsafe rather than interpreted as measurements.

`ALLOW` means the current snapshot and policy permit the workload. `DELAY` means the profile could support it but current pressure or unavailable required GPU telemetry makes it unsafe to start now. `DENY` means the profile cannot meet the requirement even when healthy.

The baseline policy reserves capacity T/TEA will not allocate: RAM is 20% of physical RAM, bounded from 512 MiB to 4 GiB; disk is 10% of total disk, bounded from 5 GiB to 50 GiB; GPU VRAM is 10% of the smallest known GPU, bounded from 128 MiB to 1 GiB (256 MiB if no GPU VRAM is known). Its GPU operational threshold is 80 C, which is a T/TEA policy threshold and not a manufacturer thermal limit.

For RAM and disk, requirement plus reserve exceeding total profile capacity is `DENY`; exceeding only currently available/free capacity is `DELAY`. A GPU workload is denied if its selected GPU is absent or lacks total VRAM, delayed if fresh GPU metrics, free VRAM, or (by default) temperature are unavailable, or if free VRAM or temperature is currently unsafe. CPU temperature is intentionally not a requirement for CPU/RAM-only workloads because it is not reliably available on all supported machines.

There is no process-name mode detection, automatic process shutdown, persistence, background polling, throttling, retrying, or workload execution. Each manager call obtains a fresh profile and snapshot. The next local-model benchmark milestone consumes `ResourceBudget` to identify candidates worth benchmarking.

Run `python -m ttea.resources` to display the current safe budget only.
