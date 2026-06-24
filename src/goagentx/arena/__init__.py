"""Arena evaluation utilities for GoAgentX."""

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    PairedEvalError,
    PairedTaskResult,
    evaluate_pair,
    make_paired_experiment_id,
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
    "SignificanceResult",
    "SignificanceTestError",
    "evaluate_pair",
    "make_paired_experiment_id",
    "permutation_test_paired_result",
    "permutation_test_score_deltas",
]
