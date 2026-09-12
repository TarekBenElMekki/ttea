# T/TEA Core Runtime Model

The `ttea.core` package supplies the smallest typed vocabulary needed to describe future runtime work: `Mission`, `Task`, `AgentSpec`, `Approval`, and `Event`. These are domain-independent records, not executable agents or orchestration logic.

Each model has a UUID4 string identifier and timezone-aware UTC timestamps. State values are `str` enums so their serialized form is stable and readable. `to_dict()` recursively converts enums and timestamps to JSON-ready primitives without writing anything to storage.

Models are deliberately mutable dataclasses. Future runtime code will need to update statuses, assignments, results, and timestamps as work progresses; lifecycle transition enforcement belongs in a later orchestration milestone. Validation is limited to required identifiers/text, sensible task attempt bounds, and UTC-aware timestamps.
