from __future__ import annotations

from datetime import datetime, timezone
import unittest

from ttea.resources import ResourceBudget
from ttea.tournament.ollama_provider import _resources_recovered, _wait_for_resource_recovery


def budget(ram: int) -> ResourceBudget:
    return ResourceBudget(
        datetime.now(timezone.utc),
        ram,
        1,
        max(0, ram - 1),
        10,
        1,
        9,
        (),
    )


class FakeManager:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def current_budget(self):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


class OllamaResourceRecoveryPollingTests(unittest.TestCase):
    def test_recovery_predicate_preserves_existing_tolerance(self):
        before = budget(4 * 1024 * 1024 * 1024)
        within = budget(before.available_ram_bytes - 200 * 1024 * 1024)
        outside = budget(before.available_ram_bytes - 300 * 1024 * 1024)
        self.assertTrue(_resources_recovered(before, within, None))
        self.assertFalse(_resources_recovered(before, outside, None))

    def test_wait_polls_until_delayed_reclamation_recovers(self):
        before = budget(4 * 1024 * 1024 * 1024)
        manager = FakeManager((
            budget(before.available_ram_bytes - 700 * 1024 * 1024),
            budget(before.available_ram_bytes - 400 * 1024 * 1024),
            budget(before.available_ram_bytes - 128 * 1024 * 1024),
        ))
        sleeps = []
        after = _wait_for_resource_recovery(
            manager,
            before,
            None,
            sleep=lambda seconds: sleeps.append(seconds),
            settle_seconds=1.0,
            attempts=5,
        )
        self.assertEqual(manager.calls, 3)
        self.assertEqual(sleeps, [1.0, 1.0, 1.0])
        self.assertTrue(_resources_recovered(before, after, None))

    def test_wait_remains_bounded_when_recovery_never_arrives(self):
        before = budget(4 * 1024 * 1024 * 1024)
        manager = FakeManager((budget(before.available_ram_bytes - 900 * 1024 * 1024),))
        sleeps = []
        after = _wait_for_resource_recovery(
            manager,
            before,
            None,
            sleep=lambda seconds: sleeps.append(seconds),
            settle_seconds=1.0,
            attempts=5,
        )
        self.assertEqual(manager.calls, 5)
        self.assertEqual(len(sleeps), 5)
        self.assertFalse(_resources_recovered(before, after, None))


if __name__ == "__main__":
    unittest.main()
