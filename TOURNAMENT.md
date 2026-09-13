# T/TEA Tournament Framework

The tournament package is an inert, provider-neutral framework for a future local-model evaluation. Execution, provisioning, downloads, inference, server lifecycle control, and process supervision remain disabled.

## Identity and manifest

A tournament uses a closed manifest: every candidate, round, and task must be declared, references must resolve, and canonical candidate identities must be unique. Candidate identity includes provider, model, quantization, parameter size, and family. The manifest has deterministic canonical JSON and a SHA-256 digest. Creation writes an immutable `manifest.json` snapshot beside `tournament.json`; load and resume reject a snapshot or current manifest whose digest differs.

A work item is identified by candidate, round, task, and repetition. Its length-prefixed stable ID is also the `TaskResult` ID, so a completed result can be skipped after restart without ambiguous string concatenation.

## Phases and sequential scheduling

The four phases are compatibility, survival, capability, and finalist. Work order is deterministic: manifest candidate order, round order, task order, then repetition. Maximum concurrency is one, so simultaneous model workloads are outside the contract.

Existing resource admission decisions map to framework actions. `ALLOW` permits conceptual progression. `DELAY` records a temporary deferral and may put the tournament in `WAITING_FOR_RESOURCES`; it never eliminates a candidate and the work remains eligible for retry. `DENY` represents permanent incompatibility and may create a resource-safety elimination. The planner never polls, sleeps, provisions, or executes.

Persistence enforces the explicit tournament state machine. Unchanged-state updates and legal transitions are accepted, while terminal states cannot become active again. `INTERRUPTED` can only resume through the explicit persisted `INTERRUPTED` to `PAUSED` operation.

## Persistence, attempts, and recovery

Persisted tournament state is typed, immutable, versioned, and explicitly serialized. Authoritative JSON replacement is atomic. Candidate states, attempts, results, eliminations, selections, cursor fields, resource-wait details, and progress survive restart. Recording the same immutable result is idempotent; different content for the same stable result ID is a conflict and is never overwritten.

Attempts have stable IDs derived from work ID and attempt number. Their lifecycle is pending, running, completed, failed, interrupted, or deferred, and persistence enforces legal transitions and timestamps. Deferred attempts have a reason but no execution timestamps. Pure crash recovery changes stale running attempts to interrupted while preserving their evidence. It never fabricates a result. A retry uses a new attempt number and completed attempts are not restarted.

Resume planning skips disabled or eliminated candidates, inapplicable rounds, and completed work. Deferred work remains schedulable. Progress preserves every valid terminal result already produced, including evidence from a candidate eliminated later; total is that evidence plus unfinished work still eligible after elimination. Future impossible work disappears without making completed progress go backwards.

## Evidence, elimination, and selection

Raw task results remain separately persisted and may only have `COMPLETED` or `FAILED` status; both are terminal task evidence and are not automatically retried. Candidate round aggregates reference and validate those result IDs, preserving intelligence dimensions and resource/performance metrics without replacing raw evidence. Metrics are deeply frozen JSON-safe values and reject non-finite numbers or arbitrary objects. Standard performance keys are `cold_load_latency_seconds`, `warm_latency_seconds`, `decode_tokens_per_second`, `peak_ram_bytes`, `peak_vram_bytes`, `gpu_temperature_c`, and `cleanup_reliability`; the first five accept finite non-negative numbers or null, GPU temperature accepts a finite number or null, and cleanup reliability accepts a boolean or 0..1 numeric value. Reports show absent values as null. Elimination records include the candidate, typed reason, location, UTC timestamp, and explanation. A safety elimination immediately makes a candidate ineligible for later selection.

Selection produces independent primary, low-resource fallback, and optional reasoning-specialist roles from eligible candidates. A composite is unavailable if any positively weighted dimension is missing; zero-weight dimensions may be absent. Reports expose score coverage and do not rank incomplete evidence. Disabled, eliminated, and validation-only candidates are excluded. `smollm2:135m-instruct-q4_0` is validation-only and can never be selected.

## Inspection

The read-only CLI can list persisted tournaments, show status, render reports, and inspect manifest metadata. `start` always refuses with `execution not enabled in framework milestone`. A compact immutable snapshot API exposes cursor, progress, wait state, latest scores, and latest elimination for a future monitor without adding a watch UI now.

## Initial 2026 Candidate Catalog

The declarative September 2026 catalog records official Ollama library tags, published package sizes, expected quantization, roles, and provisional tier-based planning capacity. Inclusion is research coverage, not endorsement: actual hardware suitability remains unknown until survival testing, published package size is not runtime RAM, and all real hardware measurements come only from tournament evidence. The first round uses 4K contexts; quantization expansion is reserved for finalists.

## Controlled Survival Round

The executable first round, `survival_4k`, proves bounded hardware and runtime compatibility at a 4K context; it does not score intelligence. Candidates are provisioned and run sequentially, one exact persisted manifest tag at a time, and missing models remain blocked unless the immutable execution-policy snapshot records explicit `--allow-model-download` authorization. All acquisition and inference use the owned `127.0.0.1:11435` runtime, while the default desktop Ollama at `127.0.0.1:11434` remains isolated and its listener identity is checked before and after owned operations.

Resource admission uses a fresh ResourceManager assessment immediately before acquisition and survival work. Temporary DELAY decisions wait without elimination on a bounded cadence and horizon, while static DENY decisions eliminate. Survival measures cold-load latency separately from decode rate, retains empirical RAM/VRAM/thermal and timing observations separately from catalog estimates, and requires unload, owned-server shutdown, runtime-state cleanup, resource recovery, thermal safety, and default-server preservation before PASS.

Detached tournament supervision persists exact PID/create-time ownership and a one-time claim token. Interrupted work can be explicitly resumed without repeating completed or eliminated candidates or redownloading a still-verified model; cancellation targets only verified owned identities. `python -m ttea.tournament watch` is a read-only live view of persisted progress, telemetry, owned PIDs, and recent events. Capability and finalist execution remain disabled.
