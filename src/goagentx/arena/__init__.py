"""Arena evaluation utilities for GoAgentX."""

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    PairedEvalError,
    PairedTaskResult,
    evaluate_pair,
    make_paired_experiment_id,
)
from goagentx.arena.runner import (
    QuickRejectDecision,
    QuickRejectError,
    QuickRejectResult,
    make_quick_reject_experiment_id,
    run_quick_reject,
    run_quick_reject_from_settings,
    select_quick_reject_tasks,
)
from goagentx.arena.stats import (
    PermutationAlternative,
    SignificanceResult,
    SignificanceTestError,
    permutation_test_paired_result,
    permutation_test_score_deltas,
)

__all__ = [
    "PairOutcome",
    "PairedEvaluationResult",
    "PairedEvalError",
    "PairedTaskResult",
    "PermutationAlternative",
    "QuickRejectDecision",
    "QuickRejectError",
    "QuickRejectResult",
    "SignificanceResult",
    "SignificanceTestError",
    "evaluate_pair",
    "make_quick_reject_experiment_id",
    "make_paired_experiment_id",
    "permutation_test_paired_result",
    "permutation_test_score_deltas",
    "run_quick_reject",
    "run_quick_reject_from_settings",
    "select_quick_reject_tasks",
]
