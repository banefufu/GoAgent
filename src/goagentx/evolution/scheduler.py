"""DreamCycle scheduling signals."""

from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.core.task import TaskRun


class StrictModel(BaseModel):
    """Base model that rejects unknown scheduler fields."""

    model_config = ConfigDict(extra="forbid")


class DegradationSettings(Protocol):
    """Settings surface needed for score-drop detection."""

    degradation_window: int
    baseline_window: int
    degradation_threshold: float


class TaskRunHistoryStore(Protocol):
    """TaskRun history surface needed by the degradation detector."""

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


class DegradationResult(StrictModel):
    """Score-drop detection result for one strategy scope."""

    strategy_id: str | None = None
    task_type: str | None = None
    triggered: bool
    reason: str
    recent_window: int = Field(..., gt=0)
    baseline_window: int = Field(..., gt=0)
    recent_sample_count: int = Field(..., ge=0)
    baseline_sample_count: int = Field(..., ge=0)
    recent_avg_score: float | None = None
    baseline_avg_score: float | None = None
    score_drop: float | None = None
    threshold: float = Field(..., ge=0.0, le=1.0)


class DegradationDetector:
    """Detect when recent strategy scores dropped versus baseline history."""

    def __init__(
        self,
        task_store: TaskRunHistoryStore,
        settings: DegradationSettings,
    ) -> None:
        """Create a detector backed by persisted TaskRun history."""
        self.task_store = task_store
        self.settings = settings

    def detect(
        self,
        *,
        strategy_id: str,
        task_type: str | None = None,
    ) -> DegradationResult:
        """Detect score degradation for a strategy and optional task type."""
        runs = self.task_store.list_recent_runs(
            limit=self.settings.degradation_window + self.settings.baseline_window,
            strategy_id=strategy_id,
            task_type=task_type,
        )
        return detect_score_degradation(
            runs,
            recent_window=self.settings.degradation_window,
            baseline_window=self.settings.baseline_window,
            threshold=self.settings.degradation_threshold,
            strategy_id=strategy_id,
            task_type=task_type,
        )


def detect_score_degradation(
    runs: Sequence[TaskRun],
    *,
    recent_window: int,
    baseline_window: int,
    threshold: float,
    strategy_id: str | None = None,
    task_type: str | None = None,
) -> DegradationResult:
    """Compare recent score history against the preceding baseline window."""
    _validate_detection_config(
        recent_window=recent_window,
        baseline_window=baseline_window,
        threshold=threshold,
    )
    run_list = list(runs)
    required_count = recent_window + baseline_window
    if len(run_list) < required_count:
        return DegradationResult(
            strategy_id=strategy_id,
            task_type=task_type,
            triggered=False,
            reason="insufficient_history",
            recent_window=recent_window,
            baseline_window=baseline_window,
            recent_sample_count=min(len(run_list), recent_window),
            baseline_sample_count=max(0, len(run_list) - recent_window),
            threshold=threshold,
        )

    recent_runs = run_list[:recent_window]
    baseline_runs = run_list[recent_window:required_count]
    recent_avg_score = _average_scores(recent_runs)
    baseline_avg_score = _average_scores(baseline_runs)
    if baseline_avg_score <= 0.0:
        return DegradationResult(
            strategy_id=strategy_id,
            task_type=task_type,
            triggered=False,
            reason="invalid_baseline_score",
            recent_window=recent_window,
            baseline_window=baseline_window,
            recent_sample_count=len(recent_runs),
            baseline_sample_count=len(baseline_runs),
            recent_avg_score=recent_avg_score,
            baseline_avg_score=baseline_avg_score,
            threshold=threshold,
        )

    score_drop = (baseline_avg_score - recent_avg_score) / baseline_avg_score
    triggered = score_drop >= threshold
    return DegradationResult(
        strategy_id=strategy_id,
        task_type=task_type,
        triggered=triggered,
        reason=(
            "score_drop_threshold_exceeded"
            if triggered
            else "score_drop_within_threshold"
        ),
        recent_window=recent_window,
        baseline_window=baseline_window,
        recent_sample_count=len(recent_runs),
        baseline_sample_count=len(baseline_runs),
        recent_avg_score=recent_avg_score,
        baseline_avg_score=baseline_avg_score,
        score_drop=score_drop,
        threshold=threshold,
    )


def _validate_detection_config(
    *,
    recent_window: int,
    baseline_window: int,
    threshold: float,
) -> None:
    """Reject invalid detector configuration."""
    if recent_window <= 0:
        raise ValueError("recent_window must be greater than 0")
    if baseline_window <= 0:
        raise ValueError("baseline_window must be greater than 0")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")


def _average_scores(runs: Sequence[TaskRun]) -> float:
    """Return average score for a non-empty TaskRun sequence."""
    return sum(run.score for run in runs) / len(runs)
