"""Arena runner workflows."""

from __future__ import annotations

import random
from collections import defaultdict
from enum import StrEnum
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.arena.paired_eval import (
    PairedEvaluationResult,
    TaskRunStore,
    evaluate_pair,
)
from goagentx.arena.stats import (
    PermutationAlternative,
    SignificanceResult,
    permutation_test_paired_result,
)
from goagentx.core.run import AgentRunner
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task


class QuickRejectError(RuntimeError):
    """Raised when quick reject cannot be run."""


class QuickRejectDecision(StrEnum):
    """Quick Reject decisions for a candidate."""

    PASS_TO_FULL_EVAL = "pass_to_full_eval"
    REJECT = "reject"


class StrictModel(BaseModel):
    """Base model that rejects unknown quick-reject fields."""

    model_config = ConfigDict(extra="forbid")


class QuickRejectResult(StrictModel):
    """Quick Reject output for one candidate."""

    decision: QuickRejectDecision
    quick_reject_passed: bool
    reason: str
    failed_checks: list[str]
    selected_task_ids: list[str] = Field(..., min_length=1)
    evaluation: PairedEvaluationResult
    regression_significance: SignificanceResult


class QuickRejectSettings(Protocol):
    """Settings surface needed by Quick Reject."""

    quick_reject_rounds: int
    min_win_rate: float
    p_value_threshold: float


def run_quick_reject(
    *,
    champion: Strategy,
    candidate: Strategy,
    tasks: Sequence[Task],
    champion_runner: AgentRunner,
    scorer: Scorer,
    candidate_runner: AgentRunner | None = None,
    rounds: int = 5,
    min_win_rate: float = 0.55,
    max_p_value: float = 0.05,
    min_score_delta: float = 0.0,
    tie_threshold: float = 0.0,
    seed: int | None = 0,
    experiment_id: str | None = None,
    task_store: TaskRunStore | None = None,
) -> QuickRejectResult:
    """Run a small stratified Arena evaluation to reject weak candidates."""
    _validate_quick_reject_config(
        rounds=rounds,
        min_win_rate=min_win_rate,
        max_p_value=max_p_value,
        tie_threshold=tie_threshold,
    )
    selected_tasks = select_quick_reject_tasks(tasks, rounds=rounds, seed=seed)
    resolved_experiment_id = experiment_id or make_quick_reject_experiment_id(
        champion_strategy_id=champion.id,
        candidate_strategy_id=candidate.id,
    )

    evaluation = evaluate_pair(
        champion=champion,
        candidate=candidate,
        tasks=selected_tasks,
        champion_runner=champion_runner,
        candidate_runner=candidate_runner,
        scorer=scorer,
        tie_threshold=tie_threshold,
        experiment_id=resolved_experiment_id,
        task_store=task_store,
    )
    regression_significance = permutation_test_paired_result(
        evaluation,
        alternative=PermutationAlternative.LESS,
        alpha=max_p_value,
        seed=seed,
    )
    failed_checks = _quick_reject_failed_checks(
        evaluation=evaluation,
        regression_significance=regression_significance,
        min_win_rate=min_win_rate,
        min_score_delta=min_score_delta,
    )
    decision = (
        QuickRejectDecision.REJECT
        if failed_checks
        else QuickRejectDecision.PASS_TO_FULL_EVAL
    )

    return QuickRejectResult(
        decision=decision,
        quick_reject_passed=decision is QuickRejectDecision.PASS_TO_FULL_EVAL,
        reason="passed_quick_reject" if not failed_checks else failed_checks[0],
        failed_checks=failed_checks,
        selected_task_ids=[task.id for task in selected_tasks],
        evaluation=evaluation,
        regression_significance=regression_significance,
    )


def run_quick_reject_from_settings(
    *,
    champion: Strategy,
    candidate: Strategy,
    tasks: Sequence[Task],
    champion_runner: AgentRunner,
    scorer: Scorer,
    settings: QuickRejectSettings,
    candidate_runner: AgentRunner | None = None,
    min_score_delta: float = 0.0,
    tie_threshold: float = 0.0,
    seed: int | None = 0,
    experiment_id: str | None = None,
    task_store: TaskRunStore | None = None,
) -> QuickRejectResult:
    """Run Quick Reject using validated arena settings."""
    return run_quick_reject(
        champion=champion,
        candidate=candidate,
        tasks=tasks,
        champion_runner=champion_runner,
        candidate_runner=candidate_runner,
        scorer=scorer,
        rounds=settings.quick_reject_rounds,
        min_win_rate=settings.min_win_rate,
        max_p_value=settings.p_value_threshold,
        min_score_delta=min_score_delta,
        tie_threshold=tie_threshold,
        seed=seed,
        experiment_id=experiment_id,
        task_store=task_store,
    )


def select_quick_reject_tasks(
    tasks: Sequence[Task],
    *,
    rounds: int,
    seed: int | None = 0,
) -> list[Task]:
    """Select a deterministic stratified sample across task buckets."""
    if rounds <= 0:
        raise QuickRejectError("rounds must be greater than 0")

    task_list = list(tasks)
    if not task_list:
        raise QuickRejectError("quick reject requires at least one task")

    grouped_tasks: dict[str, list[Task]] = defaultdict(list)
    for task in sorted(task_list, key=lambda item: (item.bucket, item.id)):
        grouped_tasks[task.bucket].append(task)

    rng = random.Random(seed)
    for bucket_tasks in grouped_tasks.values():
        rng.shuffle(bucket_tasks)

    selected_tasks: list[Task] = []
    bucket_names = sorted(grouped_tasks)
    while len(selected_tasks) < rounds:
        added_this_round = False
        for bucket in bucket_names:
            bucket_tasks = grouped_tasks[bucket]
            if not bucket_tasks:
                continue
            selected_tasks.append(bucket_tasks.pop(0))
            added_this_round = True
            if len(selected_tasks) == rounds:
                break
        if not added_this_round:
            break

    return selected_tasks


def make_quick_reject_experiment_id(
    *,
    champion_strategy_id: str,
    candidate_strategy_id: str,
) -> str:
    """Build a stable experiment id for Quick Reject runs."""
    return f"quick-reject-{champion_strategy_id}-vs-{candidate_strategy_id}"


def _validate_quick_reject_config(
    *,
    rounds: int,
    min_win_rate: float,
    max_p_value: float,
    tie_threshold: float,
) -> None:
    """Reject invalid Quick Reject settings."""
    if rounds <= 0:
        raise QuickRejectError("rounds must be greater than 0")
    if not 0 <= min_win_rate <= 1:
        raise QuickRejectError("min_win_rate must be between 0 and 1")
    if not 0 <= max_p_value <= 1:
        raise QuickRejectError("max_p_value must be between 0 and 1")
    if tie_threshold < 0:
        raise QuickRejectError("tie_threshold must be greater than or equal to 0")


def _quick_reject_failed_checks(
    *,
    evaluation: PairedEvaluationResult,
    regression_significance: SignificanceResult,
    min_win_rate: float,
    min_score_delta: float,
) -> list[str]:
    """Return failed Quick Reject guardrail names."""
    failed_checks: list[str] = []
    if evaluation.win_rate < min_win_rate:
        failed_checks.append("win_rate_below_threshold")
    if evaluation.avg_score_delta < min_score_delta:
        failed_checks.append("avg_score_delta_below_threshold")
    if (
        regression_significance.is_significant
        and regression_significance.observed_mean_delta < min_score_delta
    ):
        failed_checks.append("significant_regression")
    return failed_checks
