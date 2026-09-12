import unittest
from datetime import datetime, timedelta, timezone

from ttea.core import (
    AgentStatus,
    ApprovalStatus,
    DependencyCycleError,
    DependencyError,
    InvalidStateTransition,
    MissionStatus,
    Task,
    TaskStatus,
    can_transition,
    dependencies_completed,
    get_ready_tasks,
    is_task_blocked,
    is_task_ready,
    validate_dependencies,
    validate_transition,
)


class TransitionTests(unittest.TestCase):
    def test_intended_lifecycle_transitions_are_valid(self):
        self.assertTrue(can_transition(MissionStatus.CREATED, MissionStatus.ANALYZING))
        self.assertTrue(can_transition(MissionStatus.ANALYZING, MissionStatus.PLANNING))
        self.assertTrue(can_transition(MissionStatus.PLANNING, MissionStatus.AWAITING_APPROVAL))
        self.assertTrue(can_transition(MissionStatus.AWAITING_APPROVAL, MissionStatus.READY))
        self.assertTrue(can_transition(MissionStatus.READY, MissionStatus.RUNNING))
        self.assertTrue(can_transition(MissionStatus.RUNNING, MissionStatus.COMPLETED))
        self.assertTrue(can_transition(TaskStatus.RUNNING, TaskStatus.VERIFYING))
        self.assertTrue(can_transition(AgentStatus.PROPOSED, AgentStatus.APPROVED))
        self.assertTrue(can_transition(ApprovalStatus.PENDING, ApprovalStatus.APPROVED))

    def test_invalid_and_terminal_transitions_raise_clear_domain_error(self):
        with self.assertRaises(InvalidStateTransition):
            validate_transition(MissionStatus.CREATED, MissionStatus.RUNNING)
        with self.assertRaises(InvalidStateTransition):
            validate_transition(TaskStatus.COMPLETED, TaskStatus.RUNNING)
        with self.assertRaises(InvalidStateTransition):
            validate_transition(TaskStatus.CANCELLED, TaskStatus.READY)
        with self.assertRaises(InvalidStateTransition):
            validate_transition(ApprovalStatus.APPROVED, ApprovalStatus.PENDING)

    def test_resolved_approvals_are_terminal(self):
        for status in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.CANCELLED):
            self.assertFalse(can_transition(status, ApprovalStatus.PENDING))


class DependencyTests(unittest.TestCase):
    def task(self, task_id, *, status=TaskStatus.READY, dependencies=None, priority=0, created_at=None):
        return Task(
            id=task_id,
            mission_id="mission-1",
            title=task_id,
            objective="Test task",
            status=status,
            dependencies=dependencies or [],
            priority=priority,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def test_missing_self_and_duplicate_dependencies_are_rejected(self):
        with self.assertRaises(DependencyError):
            validate_dependencies([self.task("a", dependencies=["missing"])])
        with self.assertRaises(DependencyError):
            validate_dependencies([self.task("a", dependencies=["a"])])
        with self.assertRaises(DependencyError):
            validate_dependencies([self.task("a", dependencies=["b", "b"]), self.task("b")])

    def test_direct_and_multi_node_cycles_are_rejected(self):
        with self.assertRaises(DependencyCycleError):
            validate_dependencies([self.task("a", dependencies=["b"]), self.task("b", dependencies=["a"])])
        with self.assertRaises(DependencyCycleError):
            validate_dependencies([
                self.task("a", dependencies=["b"]),
                self.task("b", dependencies=["c"]),
                self.task("c", dependencies=["a"]),
            ])

    def test_completed_and_incomplete_dependencies_control_readiness(self):
        completed = self.task("done", status=TaskStatus.COMPLETED)
        ready = self.task("ready", dependencies=["done"])
        pending = self.task("pending", status=TaskStatus.PENDING, dependencies=["done"])
        incomplete = self.task("incomplete", dependencies=["pending"])
        tasks = [completed, ready, pending, incomplete]

        self.assertTrue(dependencies_completed(ready, tasks))
        self.assertTrue(is_task_ready(ready, tasks))
        self.assertFalse(is_task_ready(pending, tasks))
        self.assertFalse(dependencies_completed(incomplete, tasks))
        self.assertTrue(is_task_blocked(incomplete, tasks))

    def test_multiple_dependencies_and_independent_tasks(self):
        first = self.task("first", status=TaskStatus.COMPLETED)
        second = self.task("second", status=TaskStatus.COMPLETED)
        dependent = self.task("dependent", dependencies=["first", "second"])
        independent = self.task("independent")
        tasks = [first, second, dependent, independent]

        self.assertTrue(is_task_ready(dependent, tasks))
        self.assertTrue(is_task_ready(independent, tasks))

    def test_ready_task_order_is_priority_then_creation_then_id(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        low = self.task("low", priority=1, created_at=base)
        later = self.task("later", priority=5, created_at=base + timedelta(seconds=1))
        earlier = self.task("earlier", priority=5, created_at=base)
        failed = self.task("failed", status=TaskStatus.FAILED, priority=100)
        cancelled = self.task("cancelled", status=TaskStatus.CANCELLED, priority=100)

        ready_tasks = get_ready_tasks([low, later, earlier, failed, cancelled])

        self.assertEqual([task.id for task in ready_tasks], ["earlier", "later", "low"])
