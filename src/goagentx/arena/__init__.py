"""Arena evaluation utilities for GoAgentX."""

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    PairedEvalError,
    PairedTaskResult,
    evaluate_pair,
    make_paired_experiment_id,
)

__all__ = [
    "PairOutcome",
    "PairedEvaluationResult",
    "PairedEvalError",
    "PairedTaskResult",
    "evaluate_pair",
    "make_paired_experiment_id",
]
