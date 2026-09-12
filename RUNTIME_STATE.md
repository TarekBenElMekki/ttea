# T/TEA Runtime State Service

Status fields are mutable dataclass fields, but application code should change them through `ttea.core` state-service functions: `transition_mission`, `transition_task`, `transition_agent`, and `resolve_approval`.

Each operation validates the lifecycle rule first, constructs an event, then changes status and the applicable UTC timestamp. Invalid requests raise `InvalidStateTransition` before any model mutation. Successful changes return one in-memory `Event` and persist a concise informational status change through the shared logging setup.

Event types use lowercase dotted names: `mission.status_changed`, `task.status_changed`, `agent.status_changed`, and `approval.resolved`. Event payloads contain entity ID, previous and new status, and an optional supplied reason. Approval resolution also sets `resolved_at`; lifecycle validation keeps resolved approvals terminal.
