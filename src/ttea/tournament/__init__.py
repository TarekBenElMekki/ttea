"""Public contracts for T/TEA's inert local-model tournament framework."""

from .manifest import ManifestError, ManifestMismatchError, SAFETY_CANDIDATE, TournamentManifest
from .catalog import (
    CATALOG_BY_ID, CATALOG_SOURCE, CONTEXT_TARGET, INITIAL_CATALOG,
    REAL_CANDIDATES, REAL_CATALOG, RESEARCH_DATE, SAFETY_CATALOG_ENTRY,
    TIER_MEMBERSHIP, TIER_RESOURCE_ESTIMATES, CandidateCatalogEntry,
    ProvisionalResourceEstimate, ResearchTier, build_real_candidate_manifest,
    SURVIVAL_PROMPT, SURVIVAL_ROUND, SURVIVAL_TASK, build_survival_manifest,
)
from .execution import (
    ACTIVE_SURVIVAL_STATES, CandidateSurvivalRecord, ProvisioningEvidence,
    ProvisioningState, SurvivalCampaign, SurvivalCampaignState,
    SurvivalExecutionPolicy, SurvivalOutcome, SurvivalStore,
    campaign_from_dict as survival_campaign_from_dict,
    campaign_to_dict as survival_campaign_to_dict,
)
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
from .ollama_provider import (
    OllamaTournamentProvider, ProvisioningDeferred, ProvisioningRejected,
    ProvisioningResourceDelayed, ProvisioningResourceDenied,
    ProvisioningRuntimeDelayed,
    SurvivalExecutionResult,
)
from .survival import (
    SurvivalTournamentEngine, survival_report_payload, survival_report_text,
    write_survival_report,
)
from .supervisor import TournamentSupervisorManager, supervise_tournament
from .scoring import SelectionPolicy, composite, select
from .serialization import TournamentSerializationError, tournament_from_dict, tournament_to_dict
from .store import (
    AttemptConflictError, AttemptTransitionError, CandidateTransitionError,
    EvidenceConflictError, ResultConflictError, TimestampRegressionError,
    TournamentStore, TournamentStoreError, TournamentTransitionError,
)

__all__ = (
    "ACTIVE_SURVIVAL_STATES", "AttemptConflictError", "AttemptStatus", "AttemptTransitionError", "CATALOG_BY_ID", "CATALOG_SOURCE", "CONTEXT_TARGET", "CandidateCatalogEntry", "CandidateRoundResult", "CandidateState", "CandidateSurvivalRecord", "CandidateTransitionError",
    "CandidateVariant", "DIMENSIONS", "EliminationReason", "EliminationRecord", "EvidenceConflictError", "INITIAL_CATALOG", "MAX_CONCURRENCY", "PERFORMANCE_METRICS", "ProvisionalResourceEstimate",
    "ManifestError", "ManifestMismatchError", "PersistedTournament", "ProviderAvailability",
    "OllamaTournamentProvider", "ProviderMetadata", "ProvisioningDeferred", "ProvisioningEligibility", "ProvisioningEvidence", "ProvisioningRejected", "ProvisioningResourceDelayed", "ProvisioningRuntimeDelayed", "ProvisioningResourceDenied", "ProvisioningState", "REAL_CANDIDATES", "REAL_CATALOG", "RESEARCH_DATE", "ResearchTier", "ResultConflictError", "SAFETY_CATALOG_ENTRY", "SCHEMA_VERSION", "SURVIVAL_PROMPT", "SURVIVAL_ROUND", "SURVIVAL_TASK",
    "SchedulerDecision", "SelectionPolicy", "SelectionRole", "TaskResult", "TimestampRegressionError", "TournamentAttempt",
    "TournamentCandidate", "TournamentExecutionDisabled", "TournamentExecutionRequest",
    "TournamentIntegrityError", "TournamentManifest", "TournamentPhase", "TournamentProvider", "TournamentResult",
    "SurvivalCampaign", "SurvivalCampaignState", "SurvivalExecutionPolicy", "SurvivalExecutionResult", "SurvivalOutcome", "SurvivalStore", "SurvivalTournamentEngine", "TournamentRound", "TournamentSelection", "TournamentSerializationError", "TournamentSnapshot", "TournamentSupervisorManager",
    "TournamentState", "TournamentStore", "TournamentStoreError", "TournamentTask",
    "TIER_MEMBERSHIP", "TIER_RESOURCE_ESTIMATES", "TournamentTransitionError", "TournamentWorkItem", "WorkDisposition", "aggregate_round", "apply_scheduler_decision", "attempt_id", "build_real_candidate_manifest", "build_survival_manifest", "calculate_progress",
    "composite", "eliminate_candidate", "execute", "next_work", "ordered_work",
    "recover_stale_attempts", "require_attempt_transition", "require_candidate_transition", "require_transition", "retry_attempt",
    "resume_interrupted", "schedule_resource_decision", "select", "snapshot", "task_result_id",
    "supervise_tournament", "survival_campaign_from_dict", "survival_campaign_to_dict", "survival_report_payload", "survival_report_text", "tournament_from_dict", "tournament_to_dict", "valid_attempt_transition", "valid_transition", "write_survival_report",
    "validate_tournament_against_manifest", "with_calculated_progress",
)
