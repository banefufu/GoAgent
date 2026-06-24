"""Promotion gate checks for evaluated candidate strategies."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from goagentx.arena.report import FullEvaluationResult


class PromotionDecision(StrEnum):
    """Promotion gate decisions."""

    APPROVE = "approve"
    REJECT = "reject"


class PromotionGateSettings(Protocol):
    """Settings surface required by the promotion gate."""

    min_win_rate: float
    max_p_value: float
    min_score_delta: float
    max_cost_delta: float
    max_latency_delta: float
    require_no_safety_violation: bool
    require_no_critical_bucket_regression: bool


class StrictModel(BaseModel):
    """Base model that rejects unknown promotion-gate fields."""

    model_config = ConfigDict(extra="forbid")


class PromotionGateMetrics(StrictModel):
    """Metrics consumed by the promotion gate."""

    experiment_id: str = Field(..., min_length=1)
    champion_id: str = Field(..., min_length=1)
    candidate_id: str = Field(..., min_length=1)
    win_rate: float = Field(..., ge=0.0, le=1.0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_score_delta: float
    cost_delta: float
    latency_delta: float
    safety_violation_count: int | None = Field(default=None, ge=0)
    critical_bucket_regression: bool | None = None


class PromotionGateResult(StrictModel):
    """Decision and failed checks from the promotion gate."""

    decision: PromotionDecision
    approved: bool
    failed_checks: list[str]
    metrics: PromotionGateMetrics


def evaluate_promotion_gate(
    evaluation: FullEvaluationResult | PromotionGateMetrics,
    settings: PromotionGateSettings,
) -> PromotionGateResult:
    """Evaluate whether a candidate may enter promotion flow."""
    metrics = (
        promotion_metrics_from_full_eval(evaluation)
        if isinstance(evaluation, FullEvaluationResult)
        else evaluation
    )
    failed_checks = promotion_gate_failed_checks(metrics, settings)
    decision = PromotionDecision.REJECT if failed_checks else PromotionDecision.APPROVE
    return PromotionGateResult(
        decision=decision,
        approved=decision is PromotionDecision.APPROVE,
        failed_checks=failed_checks,
        metrics=metrics,
    )


def promotion_metrics_from_full_eval(
    result: FullEvaluationResult,
) -> PromotionGateMetrics:
    """Convert a Full Eval result into gate metrics."""
    return PromotionGateMetrics(
        experiment_id=result.experiment_id,
        champion_id=result.evaluation.champion_strategy_id,
        candidate_id=result.evaluation.candidate_strategy_id,
        win_rate=result.evaluation.win_rate,
        p_value=result.significance.p_value,
        avg_score_delta=result.evaluation.avg_score_delta,
        cost_delta=result.cost_delta,
        latency_delta=result.latency_delta,
        safety_violation_count=result.safety_violation_count,
        critical_bucket_regression=result.critical_bucket_regression,
    )


def promotion_gate_failed_checks(
    metrics: PromotionGateMetrics,
    settings: PromotionGateSettings,
) -> list[str]:
    """Return failed promotion-gate check names."""
    failed_checks: list[str] = []
    if metrics.win_rate < settings.min_win_rate:
        failed_checks.append("win_rate_below_threshold")
    if metrics.avg_score_delta <= settings.min_score_delta:
        failed_checks.append("avg_score_delta_below_threshold")
    if metrics.p_value is None:
        failed_checks.append("p_value_missing")
    elif metrics.p_value > settings.max_p_value:
        failed_checks.append("p_value_above_threshold")
    if metrics.cost_delta > settings.max_cost_delta:
        failed_checks.append("cost_delta_above_threshold")
    if metrics.latency_delta > settings.max_latency_delta:
        failed_checks.append("latency_delta_above_threshold")
    if settings.require_no_safety_violation:
        if metrics.safety_violation_count is None:
            failed_checks.append("safety_violation_unknown")
        elif metrics.safety_violation_count > 0:
            failed_checks.append("safety_violation")
    if settings.require_no_critical_bucket_regression:
        if metrics.critical_bucket_regression is None:
            failed_checks.append("critical_bucket_regression_unknown")
        elif metrics.critical_bucket_regression:
            failed_checks.append("critical_bucket_regression")
    return failed_checks
