from datetime import datetime, timezone
from pathlib import Path
import unittest
from uuid import uuid4

from ttea.core import AgentSpec, Approval, Event, Mission, Task
from ttea.core.enums import AgentStatus, ApprovalStatus, MissionStatus, TaskStatus
from ttea.persistence import Database, Repository
from ttea.recovery import RecoveryService, RecoveryValidationError


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("data") / f"recovery_test_{uuid4()}.db"
        self.repository = Repository(Database(self.path))
        self.repository.initialize()
        self.service = RecoveryService(self.repository)
        self.mission = Mission(id="mission-1", title="Recover", objective="Test recovery")
        self.repository.save_mission(self.mission)

    def tearDown(self):
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def task(self, task_id, status=TaskStatus.READY, dependencies=None, assigned_agent_id=None):
        return Task(id=task_id, mission_id=self.mission.id, title=task_id, objective="Recover task", status=status, dependencies=dependencies or [], assigned_agent_id=assigned_agent_id)

    def agent(self, agent_id, status):
        return AgentSpec(id=agent_id, mission_id=self.mission.id, name=agent_id, role="worker", objective="Recover agent", status=status)

    def test_missing_mission_returns_none(self):
        self.assertIsNone(self.service.recover_mission("missing"))

    def test_snapshot_reconstructs_all_models_and_classifications(self):
        completed = self.task("completed", TaskStatus.COMPLETED)
        ready = self.task("ready", dependencies=[completed.id])
        running = self.task("running", TaskStatus.RUNNING)
        pending = self.task("pending", TaskStatus.PENDING)
        blocked_ready = self.task("blocked-ready", dependencies=[pending.id])
        agent = self.agent("working-agent", AgentStatus.WORKING)
        approval = Approval(id="approval-1", mission_id=self.mission.id, approval_type="scope", requested_action="continue", reason="review")
        event = Event(id="event-1", event_type="task.status_changed", mission_id=self.mission.id, task_id=running.id, payload={"new_status": "running"})
        for task in (completed, ready, running, pending, blocked_ready):
            self.repository.save_task(task)
        self.repository.save_agent(agent)
        self.repository.save_approval(approval)
        self.repository.append_event(event)

        snapshot = self.service.recover_mission(self.mission.id)

        self.assertEqual(snapshot.mission, self.mission)
        self.assertEqual(tuple(task.id for task in snapshot.tasks), ("completed", "ready", "running", "pending", "blocked-ready"))
        self.assertEqual(snapshot.agents, (agent,))
        self.assertEqual(snapshot.approvals, (approval,))
        self.assertEqual(snapshot.events, (event,))
        self.assertEqual(snapshot.interrupted_task_ids, ("running",))
        self.assertEqual(snapshot.stale_runtime_agent_ids, ("working-agent",))
        self.assertEqual(snapshot.pending_approval_ids, ("approval-1",))
        self.assertEqual(snapshot.ready_task_ids, ("ready",))
        self.assertTrue(snapshot.requires_reconciliation)
        self.assertTrue(snapshot.awaiting_approval)

    def test_all_interrupted_task_states_are_reported_but_ready_is_stable(self):
        for task in (
            self.task("assigned", TaskStatus.ASSIGNED),
            self.task("running", TaskStatus.RUNNING),
            self.task("verifying", TaskStatus.VERIFYING),
            self.task("ready", TaskStatus.READY),
        ):
            self.repository.save_task(task)

        snapshot = self.service.recover_mission(self.mission.id)

        self.assertEqual(snapshot.interrupted_task_ids, ("assigned", "running", "verifying"))

    def test_stale_runtime_agent_states_are_reported_and_proposed_are_not(self):
        for agent_id, status in (
            ("created", AgentStatus.CREATED),
            ("idle", AgentStatus.IDLE),
            ("working", AgentStatus.WORKING),
            ("blocked", AgentStatus.BLOCKED),
            ("paused", AgentStatus.PAUSED),
            ("proposed", AgentStatus.PROPOSED),
            ("approved", AgentStatus.APPROVED),
        ):
            self.repository.save_agent(self.agent(agent_id, status))

        snapshot = self.service.recover_mission(self.mission.id)

        self.assertEqual(snapshot.stale_runtime_agent_ids, ("created", "idle", "working", "blocked", "paused"))

    def test_terminal_missions_are_excluded_from_recoverable_listing(self):
        terminal_missions = []
        for index, status in enumerate((MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED)):
            mission = Mission(id=f"terminal-{index}", title="Terminal", objective="Exclude", status=status)
            self.repository.save_mission(mission)
            terminal_missions.append(mission)

        recoverable_ids = [mission.id for mission in self.service.list_recoverable_missions()]

        self.assertEqual(recoverable_ids, [self.mission.id])
        self.assertTrue(all(mission.id not in recoverable_ids for mission in terminal_missions))

    def test_invalid_dependency_graph_raises_recovery_validation_error(self):
        self.repository.save_task(self.task("broken", dependencies=["missing"]))

        with self.assertRaisesRegex(RecoveryValidationError, "Invalid task dependencies"):
            self.service.recover_mission(self.mission.id)

    def test_missing_assigned_agent_raises_recovery_validation_error(self):
        self.repository.save_task(self.task("assigned", TaskStatus.ASSIGNED, assigned_agent_id="missing-agent"))

        with self.assertRaisesRegex(RecoveryValidationError, "missing mission agent"):
            self.service.recover_mission(self.mission.id)

    def test_recovery_does_not_mutate_persisted_state(self):
        task = self.task("running", TaskStatus.RUNNING)
        agent = self.agent("working", AgentStatus.WORKING)
        approval = Approval(id="approval", mission_id=self.mission.id, approval_type="scope", requested_action="continue", reason="review")
        self.repository.save_task(task)
        self.repository.save_agent(agent)
        self.repository.save_approval(approval)
        before = (
            self.repository.get_mission(self.mission.id).to_dict(),
            self.repository.get_task(task.id).to_dict(),
            self.repository.get_agent(agent.id).to_dict(),
            self.repository.get_approval(approval.id).to_dict(),
        )

        self.service.recover_mission(self.mission.id)

        after = (
            self.repository.get_mission(self.mission.id).to_dict(),
            self.repository.get_task(task.id).to_dict(),
            self.repository.get_agent(agent.id).to_dict(),
            self.repository.get_approval(approval.id).to_dict(),
        )
        self.assertEqual(after, before)

    def test_recovery_works_after_reopening_database(self):
        running = self.task("running", TaskStatus.RUNNING)
        ready = self.task("ready", TaskStatus.READY)
        working = self.agent("working", AgentStatus.WORKING)
        self.repository.save_task(running)
        self.repository.save_task(ready)
        self.repository.save_agent(working)

        reopened_service = RecoveryService(Repository(Database(self.path)))
        snapshot = reopened_service.recover_mission(self.mission.id)

        self.assertEqual(snapshot.interrupted_task_ids, ("running",))
        self.assertEqual(snapshot.stale_runtime_agent_ids, ("working",))
        self.assertEqual(self.repository.get_task(running.id).status, TaskStatus.RUNNING)
        self.assertEqual(self.repository.get_agent(working.id).status, AgentStatus.WORKING)

    def test_snapshot_collections_follow_repository_order(self):
        early = Task(id="early", mission_id=self.mission.id, title="Early", objective="Order", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        late = Task(id="late", mission_id=self.mission.id, title="Late", objective="Order", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.repository.save_task(late)
        self.repository.save_task(early)

        snapshot = self.service.recover_mission(self.mission.id)

        self.assertEqual(tuple(task.id for task in snapshot.tasks), ("early", "late"))
