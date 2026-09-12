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
