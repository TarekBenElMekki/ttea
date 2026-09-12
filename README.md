# T/TEA

T/TEA is currently a Python project foundation only. Mission orchestration and other domain features are intentionally out of scope for this milestone.

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
