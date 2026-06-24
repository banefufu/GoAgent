"""Evolution workflows for GoAgentX."""

from goagentx.evolution.crossover import (
    CrossoverError,
    CrossoverKind,
    CrossoverSettings,
    StrategyCrossover,
    prompt_module_crossover,
    uniform_crossover,
)
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
from goagentx.evolution.selection import (
    ParentSelectionResult,
    ParentSelectionSettings,
    ParentSelector,
    StrategyPerformance,
    select_parent_pool,
)

__all__ = [
    "CrossoverError",
    "CrossoverKind",
    "CrossoverSettings",
    "DegradationDetector",
    "DegradationResult",
    "DreamCycleAuditLogger",
    "DreamCycleCandidateResult",
    "DreamCycleError",
    "DreamCycleResult",
    "MutationError",
    "MutationKind",
    "MutationSettings",
    "ParentSelectionResult",
    "ParentSelectionSettings",
    "ParentSelector",
    "StrategyCrossover",
    "StrategyMutator",
    "StrategyPerformance",
    "detect_score_degradation",
    "load_mutation_settings",
    "prompt_module_crossover",
    "run_dreamcycle",
    "select_parent_pool",
    "uniform_crossover",
]
