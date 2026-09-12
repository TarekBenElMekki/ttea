import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch
from uuid import uuid4

from ttea.config import load_settings
from ttea.core import AgentSpec, Approval, Event, Mission, Task
from ttea.core.enums import AgentStatus, ApprovalStatus, EventSeverity, MissionStatus, TaskStatus
from ttea.persistence import Database, DuplicateEventError, Repository


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("data") / f"test_{uuid4()}.db"
        self.repository = Repository(Database(self.path))
        self.repository.initialize()
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.mission = Mission(id="mission-1", title="Persistent mission", objective="Test persistence", created_at=timestamp, updated_at=timestamp)
        self.repository.save_mission(self.mission)

    def tearDown(self):
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def test_default_and_configurable_database_paths(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Database().path, Path("data") / "ttea.db")
        with patch.dict(os.environ, {"TTEA_DATABASE_PATH": "custom/ttea.sqlite"}, clear=True):
            self.assertEqual(load_settings().database_path, Path("custom/ttea.sqlite"))

    def test_mission_round_trip_and_update(self):
        loaded = self.repository.get_mission(self.mission.id)
        self.assertEqual(loaded, self.mission)
        self.assertIsInstance(loaded.status, MissionStatus)
        self.assertEqual(loaded.created_at.tzinfo, timezone.utc)
        self.assertIsNone(self.repository.get_mission("missing"))
        self.mission.status = MissionStatus.ANALYZING
        self.mission.updated_at += timedelta(seconds=1)
        self.mission.created_at += timedelta(days=1)
        self.repository.save_mission(self.mission)
        updated = self.repository.get_mission(self.mission.id)
        self.assertEqual(updated.status, MissionStatus.ANALYZING)
        self.assertEqual(updated.updated_at, datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc))
        self.assertEqual(updated.created_at, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_task_round_trip_preserves_json_and_enum_fields(self):
        nested_time = datetime(2026, 1, 3, tzinfo=timezone.utc)
        task = Task(id="task-1", mission_id=self.mission.id, title="Persistent task", objective="Preserve task fields", status=TaskStatus.READY, priority=4, dependencies=["task-0"], assigned_agent_id="agent-1", attempts=1, max_attempts=3, result={"outcome": "partial", "scores": [1, 2], "status": ApprovalStatus.APPROVED, "time": nested_time}, artifacts=[{"name": "report", "uri": "memory://report"}])
        self.repository.save_task(task)
        loaded = self.repository.get_task(task.id)
        self.assertEqual(loaded.id, task.id)
        self.assertEqual(loaded.dependencies, task.dependencies)
        self.assertEqual(loaded.artifacts, task.artifacts)
        self.assertIsInstance(loaded.status, TaskStatus)
        self.assertEqual(loaded.result["scores"], [1, 2])
        self.assertEqual(loaded.result["status"], "approved")
        self.assertEqual(loaded.result["time"], nested_time.isoformat())
        self.assertEqual(loaded.created_at.tzinfo, timezone.utc)

    def test_agent_and_approval_round_trips(self):
        agent = AgentSpec(id="agent-1", mission_id=self.mission.id, name="Persistent agent", role="tester", objective="Preserve agent fields", status=AgentStatus.IDLE, tools=[{"name": "reader", "scopes": ["project"]}], permissions=["read:project"], model_requirements={"capabilities": ["reasoning"], "context": 4096})
        approval = Approval(id="approval-1", mission_id=self.mission.id, approval_type="scope", requested_action="continue", reason="reviewed", status=ApprovalStatus.APPROVED, resolved_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.repository.save_agent(agent)
        self.repository.save_approval(approval)
        self.assertEqual(self.repository.get_agent(agent.id), agent)
        self.assertEqual(self.repository.get_approval(approval.id), approval)
        self.assertIsInstance(self.repository.get_agent(agent.id).status, AgentStatus)
        self.assertIsInstance(self.repository.get_approval(approval.id).status, ApprovalStatus)

    def test_event_round_trip_and_duplicate_rejection(self):
        event = Event(id="event-1", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), event_type="mission.status_changed", severity=EventSeverity.WARNING, mission_id=self.mission.id, task_id="task-1", payload={"previous_status": "created", "new_status": "analyzing", "nested": {"count": 2, "severity": EventSeverity.WARNING, "time": datetime(2026, 1, 3, tzinfo=timezone.utc)}})
        self.repository.append_event(event)
        loaded = self.repository.get_event(event.id)
        self.assertEqual(loaded.id, event.id)
        self.assertEqual(loaded.timestamp, event.timestamp)
        self.assertIsInstance(loaded.severity, EventSeverity)
        self.assertEqual(loaded.payload["nested"]["severity"], "warning")
        self.assertEqual(loaded.payload["nested"]["time"], "2026-01-03T00:00:00+00:00")
        with self.assertRaises(DuplicateEventError):
            self.repository.append_event(event)

    def test_mission_isolation_and_deterministic_ordering(self):
        other = Mission(id="mission-2", title="Other", objective="Isolation", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
        early = Task(id="early", mission_id=self.mission.id, title="Early", objective="Order", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        late = Task(id="late", mission_id=self.mission.id, title="Late", objective="Order", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
        other_task = Task(id="other-task", mission_id=other.id, title="Other task", objective="Isolation")
        self.repository.save_mission(other)
        self.repository.save_task(late)
        self.repository.save_task(early)
        self.repository.save_task(other_task)
        self.assertEqual([task.id for task in self.repository.list_tasks(self.mission.id)], ["early", "late"])
        self.assertEqual([task.id for task in self.repository.list_tasks(other.id)], ["other-task"])
        self.assertEqual([mission.id for mission in self.repository.list_missions()], [self.mission.id, other.id])

    def test_foreign_keys_are_enforced(self):
        orphan = Task(mission_id="missing-mission", title="Orphan", objective="Must fail")
        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.save_task(orphan)

    def test_schema_version_is_one_after_initialization(self):
        with self.repository.database.connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, 1)

    def test_event_ordering_and_reopen_round_trip(self):
        first = Event(id="event-a", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), event_type="first", mission_id=self.mission.id, payload={})
        second = Event(id="event-b", timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), event_type="second", mission_id=self.mission.id, payload={})
        task = Task(id="restart-task", mission_id=self.mission.id, title="Restart", objective="Reopen database")
        self.repository.append_event(second)
        self.repository.append_event(first)
        self.repository.save_task(task)
        reopened = Repository(Database(self.path))
        self.assertEqual(reopened.get_mission(self.mission.id), self.mission)
        self.assertEqual(reopened.get_task(task.id), task)
        self.assertEqual([event.id for event in reopened.list_events(self.mission.id)], ["event-a", "event-b"])
