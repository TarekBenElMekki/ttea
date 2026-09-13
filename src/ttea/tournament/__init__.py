"""Public contracts for T/TEA's inert local-model tournament framework."""

from .manifest import ManifestError, ManifestMismatchError, SAFETY_CANDIDATE, TournamentManifest
from .models import (
    DIMENSIONS, MAX_CONCURRENCY, PERFORMANCE_METRICS, SCHEMA_VERSION, AttemptStatus, CandidateRoundResult,
    CandidateState, CandidateVariant, EliminationReason, EliminationRecord,
    PersistedTournament, SelectionRole, TaskResult, TournamentAttempt,
    TournamentCandidate, TournamentPhase, TournamentResult, TournamentRound,
    TournamentSelection, TournamentSnapshot, TournamentState, TournamentTask,
    TournamentWorkItem, attempt_id, recover_stale_attempts, retry_attempt,
    require_attempt_transition, valid_attempt_transition,
    require_candidate_transition, require_transition, resume_interrupted,
    snapshot, task_result_id, valid_transition,
)
from .planning import (
    SchedulerDecision, TournamentIntegrityError, WorkDisposition, aggregate_round,
    apply_scheduler_decision, calculate_progress,
    eliminate_candidate, next_work, ordered_work, schedule_resource_decision,
    validate_tournament_against_manifest, with_calculated_progress,
)
from .runner import (
    ProviderAvailability, ProviderMetadata, ProvisioningEligibility,
    TournamentExecutionDisabled, TournamentExecutionRequest,
    TournamentProvider, execute,
)
from .scoring import SelectionPolicy, composite, select
from .serialization import TournamentSerializationError, tournament_from_dict, tournament_to_dict
from .store import (
    AttemptConflictError, AttemptTransitionError, CandidateTransitionError,
    EvidenceConflictError, ResultConflictError, TimestampRegressionError,
    TournamentStore, TournamentStoreError, TournamentTransitionError,
)

__all__ = (
    "AttemptConflictError", "AttemptStatus", "AttemptTransitionError", "CandidateRoundResult", "CandidateState", "CandidateTransitionError",
    "CandidateVariant", "DIMENSIONS", "EliminationReason", "EliminationRecord", "EvidenceConflictError", "MAX_CONCURRENCY", "PERFORMANCE_METRICS",
    "ManifestError", "ManifestMismatchError", "PersistedTournament", "ProviderAvailability",
    "ProviderMetadata", "ProvisioningEligibility", "ResultConflictError", "SCHEMA_VERSION",
    "SchedulerDecision", "SelectionPolicy", "SelectionRole", "TaskResult", "TimestampRegressionError", "TournamentAttempt",
    "TournamentCandidate", "TournamentExecutionDisabled", "TournamentExecutionRequest",
    "TournamentIntegrityError", "TournamentManifest", "TournamentPhase", "TournamentProvider", "TournamentResult",
    "TournamentRound", "TournamentSelection", "TournamentSerializationError", "TournamentSnapshot",
    "TournamentState", "TournamentStore", "TournamentStoreError", "TournamentTask",
    "TournamentTransitionError", "TournamentWorkItem", "WorkDisposition", "aggregate_round", "apply_scheduler_decision", "attempt_id", "calculate_progress",
    "composite", "eliminate_candidate", "execute", "next_work", "ordered_work",
    "recover_stale_attempts", "require_attempt_transition", "require_candidate_transition", "require_transition", "retry_attempt",
    "resume_interrupted", "schedule_resource_decision", "select", "snapshot", "task_result_id",
    "tournament_from_dict", "tournament_to_dict", "valid_attempt_transition", "valid_transition",
    "validate_tournament_against_manifest", "with_calculated_progress",
)
