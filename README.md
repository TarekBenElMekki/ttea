# T/TEA

T/TEA currently provides a clean Python foundation and minimal typed runtime domain model. Mission orchestration and other execution features are intentionally out of scope for this milestone.

The current core milestone adds typed runtime primitives for missions, tasks, agent specifications, approvals, and events. See [CORE_MODEL.md](CORE_MODEL.md).

It also provides deterministic lifecycle validation and task dependency queries. See [LIFECYCLE.md](LIFECYCLE.md).

Authoritative in-memory lifecycle mutation and event creation are documented in [RUNTIME_STATE.md](RUNTIME_STATE.md).

Local SQLite persistence is documented in [PERSISTENCE.md](PERSISTENCE.md).

Read-only persisted-state analysis is documented in [RECOVERY.md](RECOVERY.md).

## Setup

Create the local environment and activate it in Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No third-party runtime dependencies are required. Run tests with the environment's Python:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Runtime logs are written to `logs/ttea.log`. Configure logging with `TTEA_LOG_LEVEL` and `TTEA_LOG_DIRECTORY`.
