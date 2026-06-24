"""Scoring utilities for task runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from goagentx.config.settings import ScoringSettings
from goagentx.core.task import TaskRun


class StrictModel(BaseModel):
    """Base model that rejects unknown scoring fields."""

    model_config = ConfigDict(extra="forbid")


class ScoringInput(StrictModel):
    """Raw metrics used to calculate a comparable task-run score."""

    quality_score: float = Field(..., ge=0.0, le=1.0)
    cost: float = Field(..., ge=0.0)
    latency_ms: int = Field(..., ge=0)
    safety_violation_count: int = Field(default=0, ge=0)


class ScoreResult(StrictModel):
    """Final score plus explainable score breakdown."""

    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float]


class Scorer:
    """Calculate stable, explainable scores from raw task-run metrics."""

    def __init__(self, settings: ScoringSettings) -> None:
        """Create a scorer from validated scoring settings."""
        self.settings = settings

    def score(self, scoring_input: ScoringInput) -> ScoreResult:
        """Calculate a score from raw metrics."""
        cost_score = _normalize_inverse(
            value=scoring_input.cost,
            max_value=self.settings.normalization.max_cost,
        )
        latency_score = _normalize_inverse(
            value=float(scoring_input.latency_ms),
            max_value=float(self.settings.normalization.max_latency_ms),
        )
        safety_score = 1.0 if scoring_input.safety_violation_count == 0 else 0.0
        safety_penalty = (
            self.settings.safety_penalty
            if scoring_input.safety_violation_count > 0
            else 0.0
        )

        weighted_total = (
            self.settings.weights.quality * scoring_input.quality_score
            + self.settings.weights.cost * cost_score
            + self.settings.weights.latency * latency_score
            + self.settings.weights.safety * safety_score
        )
        final_score = _clamp(weighted_total - safety_penalty)

        breakdown = {
            "quality": scoring_input.quality_score,
            "cost": cost_score,
            "latency": latency_score,
            "safety": safety_score,
            "safety_penalty": safety_penalty,
            "total": final_score,
        }
        return ScoreResult(score=final_score, breakdown=breakdown)

    def score_task_run(
        self,
        task_run: TaskRun,
        *,
        quality_score: float,
        safety_violation_count: int = 0,
    ) -> TaskRun:
        """Return a TaskRun copy with score and breakdown populated."""
        result = self.score(
            ScoringInput(
                quality_score=quality_score,
                cost=task_run.cost,
                latency_ms=task_run.latency_ms,
                safety_violation_count=safety_violation_count,
            )
        )
        data = task_run.model_dump(mode="python")
        data["score"] = result.score
        data["score_breakdown"] = result.breakdown
        return TaskRun.model_validate(data)


def _normalize_inverse(value: float, max_value: float) -> float:
    """Normalize lower-is-better metrics into a 0..1 score."""
    return _clamp(1.0 - value / max_value)


def _clamp(value: float) -> float:
    """Clamp a numeric value into the inclusive 0..1 range."""
    return min(1.0, max(0.0, value))
