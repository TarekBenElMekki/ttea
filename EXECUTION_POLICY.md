# T/TEA Execution Policy

LLMs and runtime components may request actions, but never grant their own permissions. `ActionRequest` is evaluated by the deterministic exact-match baseline policy: `auto` permits immediate execution; `review` requires a future deterministic review and is not permission to execute; `approval` requires an explicit human decision; and `forbidden` refuses execution.

The baseline maps `filesystem.read` to auto; `filesystem.write`, `process.execute`, and `network.read` to review; destructive, install, outbound-network, communication, and credential operations to approval; and `system.modify` to forbidden. Unmatched operations default safely to approval. Workspace- or tool-specific refinement of review is future work.

Approval helpers create a pending core `Approval` and an `approval.requested` event without persistence. The policy-to-approval bridge receives the trusted policy and re-evaluates the request before it accepts a supplied decision; only a matching approval-level decision can proceed. Approve, reject, and cancel delegate to the controlled core resolution service, producing `approval.resolved` events. Actual tool execution and automatic persistence are not implemented in this milestone.
