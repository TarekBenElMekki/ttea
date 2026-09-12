# T/TEA Hardware Telemetry

Hardware telemetry is passive and local-only: it sends no measurements externally and makes no resource-policy decisions. `psutil` provides CPU, memory, disk, uptime, and selected sensor measurements. NVIDIA GPU fields come only from local `nvidia-smi` CSV output when it is installed and working.

Unavailable metrics are represented as `None`, never zero. This is particularly important for CPU temperature: Windows commonly cannot provide a trustworthy CPU package temperature through psutil, so it remains unavailable unless a credible supported sensor is actually returned. A future LibreHardwareMonitor provider can add CPU temperature without changing the snapshot model; it is not downloaded or used now.

There is no background polling, history database, external telemetry, resource policy, or local-LLM integration in this milestone. Run `python -m ttea.telemetry` for a concise local diagnostic.
