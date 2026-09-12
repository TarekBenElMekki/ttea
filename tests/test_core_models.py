import unittest
from datetime import datetime

from ttea.core import (
    AgentSpec,
    AgentStatus,
    Approval,
    ApprovalStatus,
    Event,
    EventSeverity,
    Mission,
    MissionStatus,
    Task,
    TaskStatus,
)


class CoreModelTests(unittest.TestCase):
    def setUp(self):
        self.mission = Mission(title="Foundation", objective="Establish primitives")

    def test_models_create_with_default_states_and_unique_ids(self):
        mission_two = Mission(title="Second", objective="Confirm unique identifiers")
        task = Task(mission_id=self.mission.id, title="Model task", objective="Represent work")
        agent = AgentSpec(
            mission_id=self.mission.id,
            name="Core designer",
            role="designer",
            objective="Define primitives",
        )
        approval = Approval(
            mission_id=self.mission.id,
            approval_type="scope",
            requested_action="continue",
            reason="milestone review",
        )
        event = Event(event_type="mission_created", mission_id=self.mission.id, payload={})

        self.assertNotEqual(self.mission.id, mission_two.id)
        self.assertEqual(self.mission.status, MissionStatus.CREATED)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(agent.status, AgentStatus.PROPOSED)
        self.assertEqual(approval.status, ApprovalStatus.PENDING)
        self.assertEqual(event.severity, EventSeverity.INFO)

    def test_timestamps_are_utc_aware(self):
        for value in (self.mission.created_at, self.mission.updated_at):
            self.assertIsNotNone(value.tzinfo)
            self.assertEqual(value.utcoffset().total_seconds(), 0)

    def test_serialization_uses_predictable_enum_and_datetime_values(self):
        task = Task(
            mission_id=self.mission.id,
            title="Serialize",
            objective="Produce a dictionary",
            status=TaskStatus.READY,
            dependencies=["task-a", "task-b"],
            result={"state": ApprovalStatus.APPROVED},
        )

        serialized = task.to_dict()

        self.assertEqual(serialized["status"], "ready")
        self.assertEqual(serialized["dependencies"], ["task-a", "task-b"])
        self.assertEqual(serialized["result"]["state"], "approved")
        self.assertIsInstance(serialized["created_at"], str)
        self.assertTrue(serialized["created_at"].endswith("+00:00"))

    def test_agent_spec_and_event_payload_are_structured_and_serialized(self):
        agent = AgentSpec(
            mission_id=self.mission.id,
            name="Reviewer",
            role="reviewer",
            objective="Review output",
            tools=[{"name": "search", "scopes": ["read"]}],
            permissions=["read:project"],
            model_requirements={"capabilities": ["reasoning"], "minimum_context": 4096},
        )
        event = Event(
            event_type="agent_proposed",
            mission_id=self.mission.id,
            task_id="task-1",
            agent_id=agent.id,
            payload={"agent": agent.to_dict(), "severity": EventSeverity.WARNING},
        )

        self.assertEqual(agent.to_dict()["tools"][0]["name"], "search")
        self.assertEqual(agent.to_dict()["permissions"], ["read:project"])
        self.assertEqual(event.to_dict()["payload"]["severity"], "warning")
        self.assertEqual(event.to_dict()["payload"]["agent"]["status"], "proposed")

    def test_invalid_basic_input_is_rejected(self):
        with self.assertRaises(ValueError):
            Mission(title="", objective="Valid objective")
        with self.assertRaises(ValueError):
            Task(mission_id=self.mission.id, title="Task", objective="Work", max_attempts=0)
        with self.assertRaises(ValueError):
            Event(
                event_type="event",
                mission_id=self.mission.id,
                payload={},
                timestamp=datetime.now(),
            )
