# T/TEA Recovery

Recovery is a read-only analysis of persisted SQLite state. `RecoveryService` reconstructs one mission and its tasks, agents, approvals, and events without modifying statuses, attempts, timestamps, or database rows.

Tasks persisted as `assigned`, `running`, or `verifying` are reported as interrupted because their in-memory execution cannot survive a restart. They remain unchanged until a future reconciliation policy acts explicitly. Agents persisted as `created`, `idle`, `working`, `blocked`, or `paused` are similarly reported as stale runtime instances; this does not invalidate their persisted specifications.

Completed, failed, and cancelled missions are excluded from recoverable listings. Pending approvals are preserved and reported. Ready tasks use the existing dependency logic: they must already be `ready` and have completed dependencies.

Recovery validates mission ownership, task dependencies, and task-agent references before producing a snapshot. Structural problems raise `RecoveryValidationError`; recovery never silently repairs state. A future runtime is responsible for any reconciliation or continuation decisions.
