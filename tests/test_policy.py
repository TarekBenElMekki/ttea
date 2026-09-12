from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import unittest

from ttea.core import ApprovalStatus, InvalidStateTransition
from ttea.policy import (
    ActionRequest,
    ApprovalWorkflowError,
    ExecutionPolicyLevel,
    PolicyDecision,
    PolicyRule,
    approve_approval,
    baseline_policy,
    cancel_approval,
    reject_approval,
    request_approval,
    request_approval_for_decision,
)


class ActionRequestTests(unittest.TestCase):
    def test_request_is_valid_unique_utc_and_immutable(self):
        first = ActionRequest(mission_id="mission-1", operation_type="filesystem.read", description="Read document", metadata={"path": "notes.txt"})
        second = ActionRequest(mission_id="mission-1", operation_type="filesystem.read", description="Read document")

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.created_at.tzinfo, timezone.utc)
        with self.assertRaises(FrozenInstanceError):
            first.description = "Changed"
        with self.assertRaises(TypeError):
            first.metadata["path"] = "other.txt"

    def test_required_text_and_utc_timestamp_are_validated(self):
        with self.assertRaises(ValueError):
            ActionRequest(mission_id="", operation_type="filesystem.read", description="Read")
        with self.assertRaises(ValueError):
            ActionRequest(mission_id="mission-1", operation_type="", description="Read")
        with self.assertRaises(ValueError):
            ActionRequest(mission_id="mission-1", operation_type="filesystem.read", description="", created_at=datetime.now())

    def test_nested_metadata_is_deeply_immutable_and_independent(self):
        nested = {"paths": ["a.txt"]}
        request = ActionRequest(
            mission_id="mission-1",
            operation_type="filesystem.read",
            description="Read",
            metadata={"options": nested},
        )
        nested["paths"].append("danger.txt")

        self.assertEqual(request.metadata["options"]["paths"], ("a.txt",))
        with self.assertRaises((AttributeError, TypeError)):
            request.metadata["options"]["paths"].append("other.txt")
        with self.assertRaises(TypeError):
            request.metadata["options"]["new"] = "value"

    def test_policy_models_require_real_execution_policy_levels(self):
        with self.assertRaises(ValueError):
            PolicyRule("rule", "filesystem.read", "auto", "invalid type")
        with self.assertRaises(ValueError):
            PolicyDecision("request", "filesystem.read", "auto", "invalid type")
        with self.assertRaises(ValueError):
            PolicyDecision("", "filesystem.read", ExecutionPolicyLevel.AUTO, "missing ID")


class ExecutionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = baseline_policy()

    def request(self, operation_type):
        return ActionRequest(mission_id="mission-1", operation_type=operation_type, description="Policy test")

    def test_baseline_policy_classifications(self):
        expected = {
            "filesystem.read": ExecutionPolicyLevel.AUTO,
            "filesystem.write": ExecutionPolicyLevel.REVIEW,
            "process.execute": ExecutionPolicyLevel.REVIEW,
            "network.read": ExecutionPolicyLevel.REVIEW,
            "filesystem.delete": ExecutionPolicyLevel.APPROVAL,
            "package.install": ExecutionPolicyLevel.APPROVAL,
            "network.write": ExecutionPolicyLevel.APPROVAL,
            "communication.send": ExecutionPolicyLevel.APPROVAL,
            "credential.access": ExecutionPolicyLevel.APPROVAL,
            "system.modify": ExecutionPolicyLevel.FORBIDDEN,
        }
        for operation_type, level in expected.items():
            with self.subTest(operation_type=operation_type):
                self.assertEqual(self.policy.evaluate(self.request(operation_type)).level, level)

    def test_decision_properties_and_safe_unknown_fallback(self):
        auto = self.policy.evaluate(self.request("filesystem.read"))
        review = self.policy.evaluate(self.request("filesystem.write"))
        approval = self.policy.evaluate(self.request("communication.send"))
        forbidden = self.policy.evaluate(self.request("system.modify"))
        fallback = self.policy.evaluate(self.request("unknown.operation"))

        self.assertTrue(auto.permits_immediate_execution)
        self.assertFalse(review.permits_immediate_execution)
        self.assertTrue(review.requires_review)
        self.assertTrue(approval.requires_approval)
        self.assertTrue(forbidden.is_forbidden)
        self.assertEqual(fallback.level, ExecutionPolicyLevel.APPROVAL)
        self.assertIsNone(fallback.matched_rule_id)

    def test_evaluation_is_deterministic_and_does_not_mutate_request(self):
        request = self.request("network.read")
        before = (request.description, dict(request.metadata), request.created_at)

        first = self.policy.evaluate(request)
        second = self.policy.evaluate(request)

        self.assertEqual(first, second)
        self.assertEqual((request.description, dict(request.metadata), request.created_at), before)


class ApprovalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.policy = baseline_policy()

    def request(self, operation_type):
        return ActionRequest(mission_id="mission-1", operation_type=operation_type, description="Perform operation", target="target")

    def test_request_creates_pending_approval_and_event(self):
        approval, event = request_approval("mission-1", "Send message", "Human confirmation required")

        self.assertEqual(approval.status, ApprovalStatus.PENDING)
        self.assertEqual(event.event_type, "approval.requested")
        self.assertEqual(event.mission_id, approval.mission_id)
        self.assertEqual(event.payload["approval_id"], approval.id)
        self.assertEqual(event.payload["requested_action"], "Send message")

    def test_approval_decision_creates_execution_approval(self):
        request = self.request("communication.send")
        approval, event = request_approval_for_decision(request, self.policy.evaluate(request), self.policy)

        self.assertEqual(approval.mission_id, request.mission_id)
        self.assertEqual(approval.status, ApprovalStatus.PENDING)
        self.assertEqual(approval.approval_type, "execution")
        self.assertIn(request.operation_type, approval.requested_action)
        self.assertEqual(event.payload["approval_id"], approval.id)

    def test_non_approval_decisions_cannot_create_approval(self):
        for operation_type in ("filesystem.read", "filesystem.write", "system.modify"):
            request = self.request(operation_type)
            with self.subTest(operation_type=operation_type):
                with self.assertRaises(ApprovalWorkflowError):
                    request_approval_for_decision(request, self.policy.evaluate(request), self.policy)

    def test_trusted_policy_rejects_forged_approval_decisions(self):
        for operation_type in ("system.modify", "filesystem.read", "filesystem.write"):
            request = self.request(operation_type)
            forged = PolicyDecision(
                action_request_id=request.id,
                operation_type=request.operation_type,
                level=ExecutionPolicyLevel.APPROVAL,
                reason="forged approval",
            )
            with self.subTest(operation_type=operation_type):
                with self.assertRaises(ApprovalWorkflowError):
                    request_approval_for_decision(request, forged, self.policy)

    def test_bridge_rejects_mismatched_request_and_allows_trusted_fallback(self):
        request = self.request("communication.send")
        mismatched = PolicyDecision(
            action_request_id="different-request",
            operation_type=request.operation_type,
            level=ExecutionPolicyLevel.APPROVAL,
            reason="mismatch",
        )
        with self.assertRaises(ApprovalWorkflowError):
            request_approval_for_decision(request, mismatched, self.policy)

        unknown_request = self.request("unknown.operation")
        approval, _ = request_approval_for_decision(
            unknown_request,
            self.policy.evaluate(unknown_request),
            self.policy,
        )
        self.assertEqual(approval.status, ApprovalStatus.PENDING)

    def test_approval_resolution_helpers_reuse_core_state_service(self):
        approval, _ = request_approval("mission-1", "Approve", "Test")
        event = approve_approval(approval, "approved by reviewer")
        self.assertEqual(approval.status, ApprovalStatus.APPROVED)
        self.assertIsNotNone(approval.resolved_at)
        self.assertEqual(event.event_type, "approval.resolved")
        self.assertEqual(event.payload["new_status"], "approved")
        with self.assertRaises(InvalidStateTransition):
            approve_approval(approval)

    def test_reject_and_cancel_helpers_resolve_pending_approvals(self):
        rejected, _ = request_approval("mission-1", "Reject", "Test")
        cancelled, _ = request_approval("mission-1", "Cancel", "Test")

        reject_event = reject_approval(rejected)
        cancel_event = cancel_approval(cancelled)

        self.assertEqual(rejected.status, ApprovalStatus.REJECTED)
        self.assertEqual(cancelled.status, ApprovalStatus.CANCELLED)
        self.assertEqual(reject_event.payload["new_status"], "rejected")
        self.assertEqual(cancel_event.payload["new_status"], "cancelled")
