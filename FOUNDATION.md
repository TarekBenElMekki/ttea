# T/TEA Foundation

## Isolation

Project dependencies belong only in `.venv`; use its Python executable for project commands. The foundation uses only the Python standard library.

## Naming

The display name is T/TEA, while the repository and Python package are `ttea`. Files, modules, functions, and variables use `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`; environment variables use `TTEA_` prefixes.

## Logging

`ttea.logging` owns a single centralized setup path. It creates `logs/` automatically and writes readable, rotating runtime logs to `logs/ttea.log`; five rotated backups are retained. Obtain application loggers through `get_logger` and never log secrets.

## Structure

`src/ttea` contains the package, minimal configuration, and logging APIs. `tests` verifies the foundation. `logs/.gitkeep` retains the otherwise ignored runtime-log directory.

## Milestone decisions

Configuration is intentionally small and environment-backed. Logging is standard-library based and designed to support future named areas without a separate logging system. No domain, orchestration, provider, scheduling, database, UI, or telemetry code exists in this milestone.
