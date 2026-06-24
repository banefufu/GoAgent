"""Evolution workflows for GoAgentX."""

from goagentx.evolution.dreamcycle import (
    DreamCycleAuditLogger,
    DreamCycleCandidateResult,
    DreamCycleError,
    DreamCycleResult,
    run_dreamcycle,
)
from goagentx.evolution.mutation import (
    MutationError,
    MutationKind,
    MutationSettings,
    StrategyMutator,
    load_mutation_settings,
)
from goagentx.evolution.scheduler import (
    DegradationDetector,
    DegradationResult,
    detect_score_degradation,
)

__all__ = [
    "DegradationDetector",
    "DegradationResult",
    "DreamCycleAuditLogger",
    "DreamCycleCandidateResult",
    "DreamCycleError",
    "DreamCycleResult",
    "MutationError",
    "MutationKind",
    "MutationSettings",
    "StrategyMutator",
    "detect_score_degradation",
    "load_mutation_settings",
    "run_dreamcycle",
]
