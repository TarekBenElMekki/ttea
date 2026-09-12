import unittest
from unittest.mock import patch
import logging

from ttea.core import (
    AgentSpec,
    AgentStatus,
    Approval,
    ApprovalStatus,
    DependencyError,
    InvalidStateTransition,
    Mission,
    MissionStatus,
    Task,
    TaskStatus,
    resolve_approval,
    transition_agent,
    transition_mission,
    transition_task,
    validate_dependencies,
)


class StateServiceTests(unittest.TestCase):
    def setUp(self):
        self.mission = Mission(title="State service", objective="Mutate through a controlled API")

    def test_successful_mission_transition_updates_state_timestamp_and_event(self):
        original_updated_at = self.mission.updated_at

        event = transition_mission(self.mission, MissionStatus.ANALYZING, reason="begin review")

        self.assertEqual(self.mission.status, MissionStatus.ANALYZING)
        self.assertGreaterEqual(self.mission.updated_at, original_updated_at)
        self.assertEqual(event.event_type, "mission.status_changed")
        self.assertEqual(event.mission_id, self.mission.id)
        self.assertEqual(event.payload["previous_status"], "created")
        self.assertEqual(event.payload["new_status"], "analyzing")
        self.assertEqual(event.payload["reason"], "begin review")
        self.assertEqual(event.timestamp, self.mission.updated_at)

    def test_successful_task_and_agent_transitions_create_scoped_events(self):
        task = Task(mission_id=self.mission.id, title="Task", objective="Test state mutation")
        agent = AgentSpec(
            mission_id=self.mission.id,
            name="Agent",
            role="tester",
            objective="Test transitions",
            status=AgentStatus.CREATED,
        )
        task_updated_at = task.updated_at
        agent_updated_at = agent.updated_at

        task_event = transition_task(task, TaskStatus.READY)
        agent_event = transition_agent(agent, AgentStatus.IDLE)

        self.assertEqual(task.status, TaskStatus.READY)
        self.assertGreaterEqual(task.updated_at, task_updated_at)
        self.assertEqual(task_event.task_id, task.id)
        self.assertEqual(task_event.mission_id, self.mission.id)
        self.assertEqual(agent.status, AgentStatus.IDLE)
        self.assertGreaterEqual(agent.updated_at, agent_updated_at)
        self.assertEqual(agent_event.agent_id, agent.id)
        self.assertEqual(agent_event.payload["previous_status"], "created")
        self.assertEqual(agent_event.payload["new_status"], "idle")

    def test_approval_resolution_sets_resolved_timestamp_and_event(self):
        approval = Approval(
            mission_id=self.mission.id,
            approval_type="scope",
            requested_action="continue",
            reason="review complete",
        )

        event = resolve_approval(approval, ApprovalStatus.APPROVED)

        self.assertEqual(approval.status, ApprovalStatus.APPROVED)
        self.assertIsNotNone(approval.resolved_at)
        self.assertEqual(event.event_type, "approval.resolved")
        self.assertEqual(event.payload["approval_id"], approval.id)
        self.assertEqual(event.payload["previous_status"], "pending")
        self.assertEqual(event.payload["new_status"], "approved")
        self.assertEqual(event.timestamp, approval.resolved_at)

    def test_invalid_transition_is_atomic_and_returns_no_success_event(self):
        original_status = self.mission.status
        original_updated_at = self.mission.updated_at

        with self.assertRaises(InvalidStateTransition):
            transition_mission(self.mission, MissionStatus.RUNNING)

        self.assertEqual(self.mission.status, original_status)
        self.assertEqual(self.mission.updated_at, original_updated_at)

    def test_logging_setup_failure_happens_before_mutation(self):
        root_logger = logging.getLogger("ttea")
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()
        original_status = self.mission.status
        original_updated_at = self.mission.updated_at
        try:
            with patch("ttea.core.state_service.setup_logging", side_effect=OSError("logging unavailable")):
                with self.assertRaises(OSError):
                    transition_mission(self.mission, MissionStatus.ANALYZING)
        finally:
            root_logger.handlers[:] = original_handlers

        self.assertEqual(self.mission.status, original_status)
        self.assertEqual(self.mission.updated_at, original_updated_at)

    def test_resolved_approval_remains_unchanged_after_invalid_resolution(self):
        approval = Approval(
            mission_id=self.mission.id,
            approval_type="scope",
            requested_action="continue",
            reason="review complete",
        )
        resolve_approval(approval, ApprovalStatus.REJECTED)
        resolved_at = approval.resolved_at

        with self.assertRaises(InvalidStateTransition):
            resolve_approval(approval, ApprovalStatus.APPROVED)

        self.assertEqual(approval.status, ApprovalStatus.REJECTED)
        self.assertEqual(approval.resolved_at, resolved_at)

    def test_cross_mission_dependency_graph_is_rejected(self):
        other_mission = Mission(title="Other", objective="Provide another ID")
        first = Task(mission_id=self.mission.id, title="First", objective="First task")
        second = Task(
            mission_id=other_mission.id,
            title="Second",
            objective="Second task",
            dependencies=[first.id],
        )

        with self.assertRaisesRegex(DependencyError, "same mission"):
            validate_dependencies([first, second])
