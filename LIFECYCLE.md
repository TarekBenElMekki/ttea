# T/TEA Lifecycle and Dependencies

Core lifecycle rules are pure validators: `can_transition()` answers a transition query and `validate_transition()` raises `InvalidStateTransition` when it is invalid. Neither changes model status or timestamps. Terminal mission states are completed, failed, and cancelled; completed and cancelled tasks are terminal; terminated agents and resolved approvals are terminal. A failed task may return to ready only for a future retry policy to use deliberately.

`validate_dependencies()` rejects duplicate task IDs, missing references, self-references, duplicate references, and cycles. `dependencies_completed()` requires every referenced task to be completed. A task is ready only when its status is `ready` and all dependencies are complete; a blocked task or one with incomplete dependencies is reported blocked. These functions never mutate task status.

`get_ready_tasks()` validates the complete graph first, then returns eligible tasks ordered by higher numeric priority, earlier creation time, and ID as a deterministic final tie-breaker.
