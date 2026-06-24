"""Arena evaluation utilities for GoAgentX."""

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    PairedEvalError,
    PairedTaskResult,
    evaluate_pair,
    make_paired_experiment_id,
)
from goagentx.arena.report import (
    FullEvalError,
    FullEvaluationResult,
    FullEvalVerdict,
    full_eval_result_to_experiment,
    make_full_eval_experiment_id,
    render_full_eval_report,
    run_full_eval,
    run_full_eval_from_settings,
    select_full_eval_tasks,
    write_full_eval_report,
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
    "FullEvalError",
    "FullEvaluationResult",
    "FullEvalVerdict",
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
    "full_eval_result_to_experiment",
    "make_full_eval_experiment_id",
    "make_quick_reject_experiment_id",
    "make_paired_experiment_id",
    "permutation_test_paired_result",
    "permutation_test_score_deltas",
    "render_full_eval_report",
    "run_full_eval",
    "run_full_eval_from_settings",
    "run_quick_reject",
    "run_quick_reject_from_settings",
    "select_full_eval_tasks",
    "select_quick_reject_tasks",
    "write_full_eval_report",
]
