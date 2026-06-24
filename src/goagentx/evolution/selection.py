"""Parent-pool selection for genome GA."""

from __future__ import annotations

from math import ceil
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.core.strategy import Strategy
from goagentx.core.task import TaskRun


class StrictModel(BaseModel):
    """Base model that rejects unknown selection fields."""

    model_config = ConfigDict(extra="forbid")


class TaskRunHistoryStore(Protocol):
    """TaskRun history surface needed by parent selection."""

    def list_recent_runs(
        self,
        *,
        limit: int = 10,
        task_id: str | None = None,
        strategy_id: str | None = None,
        experiment_id: str | None = None,
        task_type: str | None = None,
    ) -> list[TaskRun]:
        """Return recent TaskRuns, newest first."""


class ParentSelectionSettings(StrictModel):
    """Configurable parent selection bounds."""

    selection_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    min_parent_count: int = Field(default=2, ge=1)
    elite_count: int = Field(default=1, ge=0)
    max_runs_per_strategy: int = Field(default=50, gt=0)


class StrategyPerformance(StrictModel):
    """Historical score summary for one strategy."""

    strategy: Strategy
    average_score: float
    run_count: int = Field(..., gt=0)


class ParentSelectionResult(StrictModel):
    """Selected GA parents plus the full ranked performance table."""

    task_type: str | None = None
    selection_ratio: float
    min_parent_count: int
    elite_count: int
    ranked_performances: list[StrategyPerformance]
    parent_pool: list[Strategy]
    elite_pool: list[Strategy]
    insufficient_parent_pool: bool
    reason: str

    @property
    def parent_ids(self) -> list[str]:
        """Return selected parent ids in rank order."""
        return [strategy.id for strategy in self.parent_pool]

    @property
    def elite_ids(self) -> list[str]:
        """Return selected elite ids in rank order."""
        return [strategy.id for strategy in self.elite_pool]


class ParentSelector:
    """Select parent strategies from historical task-run scores."""

    def __init__(
        self,
        task_store: TaskRunHistoryStore,
        settings: ParentSelectionSettings | None = None,
    ) -> None:
        """Create a selector backed by TaskRun history."""
        self.task_store = task_store
        self.settings = settings or ParentSelectionSettings()

    def select(
        self,
        strategies: Sequence[Strategy],
        *,
        task_type: str | None = None,
    ) -> ParentSelectionResult:
        """Rank strategies by average score and return the parent pool."""
        ranked_performances = _rank_performances(
            strategies,
            self.task_store,
            task_type=task_type,
            max_runs_per_strategy=self.settings.max_runs_per_strategy,
        )
        selected_count = _selection_count(
            ranked_count=len(ranked_performances),
            settings=self.settings,
        )
        parent_pool = [
            performance.strategy for performance in ranked_performances[:selected_count]
        ]
        elite_count = min(self.settings.elite_count, len(ranked_performances))
        elite_pool = [
            performance.strategy for performance in ranked_performances[:elite_count]
        ]
        insufficient_parent_pool = len(parent_pool) < self.settings.min_parent_count

        return ParentSelectionResult(
            task_type=task_type,
            selection_ratio=self.settings.selection_ratio,
            min_parent_count=self.settings.min_parent_count,
            elite_count=self.settings.elite_count,
            ranked_performances=ranked_performances,
            parent_pool=parent_pool,
            elite_pool=elite_pool,
            insufficient_parent_pool=insufficient_parent_pool,
            reason=_selection_reason(
                ranked_count=len(ranked_performances),
                insufficient_parent_pool=insufficient_parent_pool,
            ),
        )


def select_parent_pool(
    strategies: Sequence[Strategy],
    task_store: TaskRunHistoryStore,
    *,
    task_type: str | None = None,
    selection_ratio: float = 0.6,
    min_parent_count: int = 2,
    elite_count: int = 1,
    max_runs_per_strategy: int = 50,
) -> ParentSelectionResult:
    """Select a GA parent pool using historical average scores."""
    settings = ParentSelectionSettings(
        selection_ratio=selection_ratio,
        min_parent_count=min_parent_count,
        elite_count=elite_count,
        max_runs_per_strategy=max_runs_per_strategy,
    )
    return ParentSelector(task_store, settings).select(
        strategies,
        task_type=task_type,
    )


def _rank_performances(
    strategies: Sequence[Strategy],
    task_store: TaskRunHistoryStore,
    *,
    task_type: str | None,
    max_runs_per_strategy: int,
) -> list[StrategyPerformance]:
    performances: list[StrategyPerformance] = []
    for strategy in strategies:
        if not _strategy_matches_task_type(strategy, task_type):
            continue
        performance = _strategy_performance(
            strategy,
            task_store,
            task_type=task_type,
            max_runs_per_strategy=max_runs_per_strategy,
        )
        if performance is not None:
            performances.append(performance)

    return sorted(
        performances,
        key=lambda performance: (
            -performance.average_score,
            -performance.run_count,
            performance.strategy.id,
        ),
    )


def _strategy_matches_task_type(strategy: Strategy, task_type: str | None) -> bool:
    if task_type is None:
        return True
    return strategy.task_type is None or strategy.task_type == task_type


def _strategy_performance(
    strategy: Strategy,
    task_store: TaskRunHistoryStore,
    *,
    task_type: str | None,
    max_runs_per_strategy: int,
) -> StrategyPerformance | None:
    runs = task_store.list_recent_runs(
        limit=max_runs_per_strategy,
        strategy_id=strategy.id,
        task_type=task_type,
    )
    if not runs:
        return None
    return StrategyPerformance(
        strategy=strategy,
        average_score=sum(run.score for run in runs) / len(runs),
        run_count=len(runs),
    )


def _selection_count(
    *,
    ranked_count: int,
    settings: ParentSelectionSettings,
) -> int:
    if ranked_count == 0:
        return 0
    desired_count = ceil(ranked_count * settings.selection_ratio)
    desired_count = max(
        desired_count,
        settings.min_parent_count,
        settings.elite_count,
    )
    return min(desired_count, ranked_count)


def _selection_reason(
    *,
    ranked_count: int,
    insufficient_parent_pool: bool,
) -> str:
    if ranked_count == 0:
        return "no_scored_strategies"
    if insufficient_parent_pool:
        return "insufficient_scored_strategies"
    return "selected"
