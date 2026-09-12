# Benchmark Safety Supervisor

Zero-risk benchmarking cannot literally be guaranteed, so future local-model tests are bounded and independently supervised. A parent supervisor requires an `ALLOW` Resource Manager preflight, starts exactly one child worker for one model and one request, and samples fresh runtime telemetry while the child runs.

Preflight requires the requested GPU allocation plus reserve to fit before launch. Once resident, the watchdog protects the remaining VRAM reserve without double-counting that allocation. It aborts after consecutive RAM or VRAM reserve violations, immediately aborts when required GPU telemetry disappears, and uses the stricter of the benchmark and effective Resource Policy thermal thresholds. GPU warning temperatures are logged. CPU temperature is not used because it is not reliably available on all supported machines.

Context, output, CPU threads, individual requests, and whole-model runtime are policy-bounded. Every exit path—including completion, timeout, pressure, thermal cutoff, interruption, or crash—terminates only the owned child, confirms termination, and invokes an idempotent cleanup hierarchy whose individual steps have finite timeouts. Future cleanup may attempt API unload, `ollama stop`, then termination of a benchmark-owned process tree; no Ollama integration exists yet.

Active worker PID and creation time plus supervisor PID, creation time, and start time are kept temporarily in ignored `data/runtime/benchmark.json`. Supervisor identity protects the child-startup window. State is removed only after worker shutdown is confirmed; otherwise it remains for the manual kill switch. `scripts/Stop-TTeaBenchmark.ps1` stops only the recorded PID after its process creation time matches; absent, invalid, stale, or mismatched state is harmless.

Run `python -m ttea.benchmarking --self-test` before any future real LLM execution. The self-test creates only harmless no-op and sleeping children and synthetic telemetry; it performs no model call or stress workload.
