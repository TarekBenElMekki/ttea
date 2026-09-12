# T/TEA Persistence

T/TEA uses the Python standard library's SQLite support for a local, single-process runtime. The default database is `data/ttea.db`; set `TTEA_DATABASE_PATH` to override it. Call `Database.initialize()` or `Repository.initialize()` explicitly before repository use.

The schema stores core missions, tasks, agent specifications, approvals, and events. Child records use restrictive foreign keys to keep mission history from being silently deleted. Structured fields are JSON text and timestamps are stored as ISO-8601 UTC strings.

Schema versioning currently uses SQLite `PRAGMA user_version`; newly initialized databases are version `1`. Migrations are not implemented yet.

Mutable core records use upserts. Events are append-only: an existing event ID raises `DuplicateEventError`, including if its content differs. SQLite is local durability, not a distributed or pooled-connection design; recovery consists of reopening the database and loading records through the repository.
