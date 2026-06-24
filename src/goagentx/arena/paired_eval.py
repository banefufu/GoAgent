"""Paired strategy evaluation for Arena."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.core.run import AgentRunner, run_agent_task
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task, TaskRun


class PairedEvalError(RuntimeError):
    """Raised when a paired evaluation cannot be run."""


class PairOutcome(StrEnum):
    """Candidate outcome for one paired task comparison."""

    WIN = "win"
    LOSS = "loss"
    TIE = "tie"


class StrictModel(BaseModel):
    """Base model that rejects unknown paired-eval fields."""

    model_config = ConfigDict(extra="forbid")


class PairedTaskResult(StrictModel):
    """Candidate-vs-champion result for one task."""

    task_id: str
    task_type: str
    bucket: str
    champion_run: TaskRun
    candidate_run: TaskRun
    score_delta: float
    outcome: PairOutcome


class PairedEvaluationResult(StrictModel):
    """Aggregate paired-evaluation result for one candidate."""

    experiment_id: str
    champion_strategy_id: str
    candidate_strategy_id: str
    task_count: int = Field(..., gt=0)
    wins: int = Field(..., ge=0)
    losses: int = Field(..., ge=0)
    ties: int = Field(..., ge=0)
    win_rate: float = Field(..., ge=0.0, le=1.0)
    avg_score_delta: float
    results: list[PairedTaskResult] = Field(..., min_length=1)


class TaskRunStore(Protocol):
    """Persistence surface needed by paired evaluation."""

    def save_run(self, task_run: TaskRun) -> TaskRun:
        """Persist one TaskRun."""


def evaluate_pair(
    *,
    champion: Strategy,
    candidate: Strategy,
    tasks: Sequence[Task],
    champion_runner: AgentRunner,
    scorer: Scorer,
    candidate_runner: AgentRunner | None = None,
    tie_threshold: float = 0.0,
    experiment_id: str | None = None,
    task_store: TaskRunStore | None = None,
) -> PairedEvaluationResult:
    """Evaluate champion and candidate on the same tasks."""
    task_list = list(tasks)
    if not task_list:
        raise PairedEvalError("paired evaluation requires at least one task")
    if tie_threshold < 0:
        raise PairedEvalError("tie_threshold must be greater than or equal to 0")

    resolved_candidate_runner = candidate_runner or champion_runner
    resolved_experiment_id = experiment_id or make_paired_experiment_id(
        champion_strategy_id=champion.id,
        candidate_strategy_id=candidate.id,
    )

    results: list[PairedTaskResult] = []
    for task in task_list:
        champion_run = run_agent_task(
            strategy=champion,
            task=task,
            runner=champion_runner,
            scorer=scorer,
            experiment_id=resolved_experiment_id,
        )
        candidate_run = run_agent_task(
            strategy=candidate,
            task=task,
            runner=resolved_candidate_runner,
            scorer=scorer,
            experiment_id=resolved_experiment_id,
        )
        if task_store is not None:
            task_store.save_run(champion_run)
            task_store.save_run(candidate_run)

        score_delta = candidate_run.score - champion_run.score
        results.append(
            PairedTaskResult(
                task_id=task.id,
                task_type=task.task_type,
                bucket=task.bucket,
                champion_run=champion_run,
                candidate_run=candidate_run,
                score_delta=score_delta,
                outcome=_outcome(score_delta, tie_threshold=tie_threshold),
            )
        )

    wins = _count_outcomes(results, PairOutcome.WIN)
    losses = _count_outcomes(results, PairOutcome.LOSS)
    ties = _count_outcomes(results, PairOutcome.TIE)
    task_count = len(results)
    avg_score_delta = sum(result.score_delta for result in results) / task_count

    return PairedEvaluationResult(
        experiment_id=resolved_experiment_id,
        champion_strategy_id=champion.id,
        candidate_strategy_id=candidate.id,
        task_count=task_count,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=wins / task_count,
        avg_score_delta=avg_score_delta,
        results=results,
    )


def make_paired_experiment_id(
    *,
    champion_strategy_id: str,
    candidate_strategy_id: str,
) -> str:
    """Build a stable Arena experiment id for a strategy pair."""
    return f"eval-{champion_strategy_id}-vs-{candidate_strategy_id}"


def _outcome(score_delta: float, *, tie_threshold: float) -> PairOutcome:
    """Classify one candidate score delta."""
    if score_delta > tie_threshold:
        return PairOutcome.WIN
    if score_delta < -tie_threshold:
        return PairOutcome.LOSS
    return PairOutcome.TIE


def _count_outcomes(
    results: Sequence[PairedTaskResult],
    outcome: PairOutcome,
) -> int:
    """Count task results with a specific outcome."""
    return sum(1 for result in results if result.outcome == outcome)
