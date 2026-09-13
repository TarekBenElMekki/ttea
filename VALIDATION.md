# Autonomous system validation

T/TEA validation is real-machine integration verification: unit tests remain deterministic and mocked, while benchmarks own bounded model execution. Validation orchestrates safe system checks without downloading models or running inference.

Run the pre-model suite unattended with:

```powershell
.\.venv\Scripts\python.exe -m ttea.validation run --suite pre_model
```

The runner performs verified stale-state reclamation, resource preflight, loopback-port checks, model-less owned-server lifecycle checks, missing-model refusal, default-server isolation, tolerant resource recovery, and an exact-owned-process emergency recovery exercise. It never kills by process name, modifies services or PATH, closes user applications, or controls the default Ollama server.

Scenario states have distinct meanings: `PASS` verifies the invariant, `FAIL` observes a violated invariant or unverified cleanup, `BLOCKED` means a safe precondition prevented execution, and `SKIP` means the scenario was intentionally inapplicable. A blocked resource precondition can be retried for a bounded period with `--wait-for-resources SECONDS`; the runner never waits indefinitely or manufactures capacity by terminating applications.

Each run writes `report.json` and `report.txt` under `data/validation/<run-id>/`. Use `python -m ttea.validation list` to list suites, `run --suite pre_model --json` for machine-readable console output, and `python -m ttea.validation latest` to display the newest saved report without rerunning checks.

Exit codes are stable: `0` PASS, `1` FAIL, `2` BLOCKED, and `130` interrupted. Cleanup verification is mandatory for success; if cleanup cannot be proven, the run fails and ownership state is retained for recovery.

The model-less recovery tolerance is `max(256 MiB, 5% of baseline available RAM)` and, per GPU, `max(64 MiB, 5% of baseline free VRAM)`. Reports record before/after values and deltas because normal Windows caching makes byte-for-byte restoration unreliable.

The suite registry is intentionally extensible for future `tiny_model`, `model_tournament`, `runtime_regression`, and `full_system` suites. None of those model-running suites is implemented yet.

Process control always requires an exact trusted handle or verified PID plus creation time. The PowerShell emergency-stop script remains a disaster-recovery backup, not part of routine validation.

## Persistent campaigns

A campaign persistently advances through registered validation suites and requires `pre_model` to reach a real `PASS` before any future model suite can run. Start the current campaign detached from the launching terminal:

```powershell
.\.venv\Scripts\python.exe -m ttea.validation campaign start --through pre_model --detach
```

The lightweight detached Python supervisor uses no Windows service, Task Scheduler entry, administrator privilege, or second terminal. On a resource-only `DELAY`, it records `WAITING_FOR_RESOURCES`, sleeps for a bounded polling interval (30 seconds by default), samples fresh telemetry, and retries automatically. It never detects development tools by process name, closes applications, or waits forever; non-resource blocks and `DENY` become `BLOCKED`, validation failures become `FAILED`, and the default maximum duration is six hours.

Use `campaign status` (or `campaign status --json`) to inspect the persisted heartbeat, last validation, reason, retry time, and verified supervisor identity. `campaign list` shows campaign history, `campaign cancel` requests graceful shutdown before considering exact PID/create-time termination, and `campaign resume` safely restarts an interrupted or blocked campaign only after the benchmark ownership guard accepts the machine state.

Campaign state and its low-frequency log live under `data/validation/campaigns/<campaign-id>/`. Atomic `campaign.json` updates contain validation run references and resource-wait periods; terminal campaigns also produce `campaign-report.json` and `campaign-report.txt` without duplicating suite reports. A crashed or stale supervisor is marked `INTERRUPTED` by status inspection and remains resumable rather than having its ownership data blindly deleted.

Routine campaigns rely on the same resource, process, port, and cleanup guards as direct validation. The emergency PowerShell script remains fallback recovery only.

## Tiny-model inference safety suite

The `tiny_model` suite validates controlled provisioning, one bounded inference, forced outer-timeout cleanup, and resource recovery. Its only allow-listed asset is `smollm2:135m-instruct-q4_0`, used strictly for safety validation and never treated as an intelligence benchmark candidate; reports identify the installed artifact by digest and reported size as well as name.

Benchmark execution still never pulls models. Acquisition exists only in the validation asset layer, uses a T/TEA-owned Ollama server on loopback port 11435, and posts the exact allow-listed model to the bounded non-streaming local pull API. A campaign may acquire it only when started with explicit persistent authorization:

```powershell
.\.venv\Scripts\python.exe -m ttea.validation campaign start --through tiny_model --allow-model-download --detach
```

## Campaign monitor

Use the native read-only monitor to inspect the current or latest persisted validation campaign:

```powershell
.\.venv\Scripts\python.exe -m ttea.validation campaign watch
.\.venv\Scripts\python.exe -m ttea.validation campaign watch --refresh 1
.\.venv\Scripts\python.exe -m ttea.validation campaign watch --once
.\.venv\Scripts\python.exe -m ttea.validation campaign watch --no-color
```

The watch command is read-only. Closing it does not stop the detached validation campaign.

Without `--allow-model-download`, an already-installed exact model may be validated but a missing model blocks the suite. Download preflight separately checks disk and model-less server resources, allowing provisioning to finish even when the later 512 MiB inference estimate remains delayed. Completed provisioning is retained, so campaign retries inspect the local digest instead of downloading repeatedly.

Inference follows `ResourceManager -> BenchmarkSupervisor -> OllamaBenchmarkWorker` with at most 512 context tokens, 16 generated tokens, two CPU threads, non-streaming output, and zero keep-alive. The forced-timeout scenario uses a much shorter outer model deadline without creating an unbounded request; if both bounded attempts finish too quickly, that scenario is reported as skipped rather than failed.

The suite records Ollama timing/token metrics and derives tokens per second when possible. Existing telemetry cadence captures RAM and VRAM minima plus maximum GPU temperature. Success requires exact-model unload verification, owned process-tree shutdown, port and runtime-state cleanup, preservation of the default server identity, and RAM/VRAM recovery within the documented validation tolerances.
