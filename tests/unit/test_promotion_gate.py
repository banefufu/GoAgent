from pathlib import Path

import pytest

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    PairedTaskResult,
)
from goagentx.arena.report import FullEvaluationResult, FullEvalVerdict
from goagentx.arena.stats import PermutationAlternative, SignificanceResult
from goagentx.config.settings import load_settings
from goagentx.core.task import TaskRun
from goagentx.promotion.gate import (
    PromotionDecision,
    PromotionGateMetrics,
    evaluate_promotion_gate,
    promotion_metrics_from_full_eval,
)


def test_promotion_gate_approves_candidate_that_meets_all_thresholds() -> None:
    result = evaluate_promotion_gate(_metrics(), load_settings().promotion_gate)

    assert result.approved is True
    assert result.decision is PromotionDecision.APPROVE
    assert result.failed_checks == []


def test_promotion_gate_rejects_any_hard_threshold_failure() -> None:
    result = evaluate_promotion_gate(
        _metrics(
            win_rate=0.5,
            p_value=0.2,
            avg_score_delta=0.0,
            cost_delta=0.25,
            latency_delta=0.3,
            safety_violation_count=1,
            critical_bucket_regression=True,
        ),
        load_settings().promotion_gate,
    )

    assert result.approved is False
    assert result.decision is PromotionDecision.REJECT
    assert result.failed_checks == [
        "win_rate_below_threshold",
        "avg_score_delta_below_threshold",
        "p_value_above_threshold",
        "cost_delta_above_threshold",
        "latency_delta_above_threshold",
        "safety_violation",
        "critical_bucket_regression",
    ]


def test_missing_p_value_rejects_promotion() -> None:
    result = evaluate_promotion_gate(
        _metrics(p_value=None),
        load_settings().promotion_gate,
    )

    assert result.approved is False
    assert result.failed_checks == ["p_value_missing"]


def test_missing_safety_or_critical_metrics_reject_when_required() -> None:
    result = evaluate_promotion_gate(
        _metrics(
            safety_violation_count=None,
            critical_bucket_regression=None,
        ),
        load_settings().promotion_gate,
    )

    assert result.approved is False
    assert result.failed_checks == [
        "safety_violation_unknown",
        "critical_bucket_regression_unknown",
    ]


def test_optional_safety_and_critical_checks_can_be_disabled() -> None:
    settings = load_settings().promotion_gate.model_copy(
        update={
            "require_no_safety_violation": False,
            "require_no_critical_bucket_regression": False,
        }
    )

    result = evaluate_promotion_gate(
        _metrics(
            safety_violation_count=None,
            critical_bucket_regression=None,
        ),
        settings,
    )

    assert result.approved is True
    assert result.failed_checks == []


def test_promotion_gate_accepts_full_eval_result() -> None:
    full_eval_result = _full_eval_result()

    result = evaluate_promotion_gate(full_eval_result, load_settings().promotion_gate)

    assert result.approved is True
    assert result.metrics.experiment_id == "full-eval-champion-vs-candidate"
    assert result.metrics.champion_id == "champion"
    assert result.metrics.candidate_id == "candidate"
    assert result.metrics.safety_violation_count == 0
    assert result.metrics.critical_bucket_regression is False


def test_full_eval_conversion_preserves_gate_metrics() -> None:
    metrics = promotion_metrics_from_full_eval(_full_eval_result())

    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.p_value == pytest.approx(0.01)
    assert metrics.avg_score_delta == pytest.approx(0.1)
    assert metrics.cost_delta == pytest.approx(0.05)
    assert metrics.latency_delta == pytest.approx(0.05)


def _metrics(**overrides: object) -> PromotionGateMetrics:
    data = {
        "experiment_id": "full-eval-champion-vs-candidate",
        "champion_id": "champion",
        "candidate_id": "candidate",
        "win_rate": 0.8,
        "p_value": 0.01,
        "avg_score_delta": 0.1,
        "cost_delta": 0.05,
        "latency_delta": 0.05,
        "safety_violation_count": 0,
        "critical_bucket_regression": False,
    }
    data.update(overrides)
    return PromotionGateMetrics.model_validate(data)


def _full_eval_result() -> FullEvaluationResult:
    champion_run = _task_run("run-champion", "champion", score=0.8)
    candidate_run = _task_run("run-candidate", "candidate", score=0.9)
    task_result = PairedTaskResult(
        task_id="task-001",
        task_type="doc_qa",
        bucket="baseline",
        champion_run=champion_run,
        candidate_run=candidate_run,
        score_delta=0.1,
        outcome=PairOutcome.WIN,
    )
    evaluation = PairedEvaluationResult(
        experiment_id="full-eval-champion-vs-candidate",
        champion_strategy_id="champion",
        candidate_strategy_id="candidate",
        task_count=1,
        wins=1,
        losses=0,
        ties=0,
        win_rate=1.0,
        avg_score_delta=0.1,
        results=[task_result],
    )
    significance = SignificanceResult(
        alternative=PermutationAlternative.GREATER,
        sample_count=8,
        effective_sample_count=8,
        observed_mean_delta=0.1,
        p_value=0.01,
        alpha=0.05,
        is_significant=True,
        insufficient_sample=False,
        permutations=256,
        seed=0,
    )
    return FullEvaluationResult(
        experiment_id="full-eval-champion-vs-candidate",
        task_set_id="test-set",
        verdict=FullEvalVerdict.PROMOTE_READY,
        full_eval_passed=True,
        failed_checks=[],
        selected_task_ids=["task-001"],
        evaluation=evaluation,
        significance=significance,
        cost_delta=0.05,
        latency_delta=0.05,
        safety_violation_count=0,
        critical_bucket_regression=False,
        report_path=Path("reports/full-eval.md"),
    )


def _task_run(run_id: str, strategy_id: str, *, score: float) -> TaskRun:
    return TaskRun(
        id=run_id,
        task_id="task-001",
        strategy_id=strategy_id,
        output_json={"answer": "GoAgentX evolves agent strategies."},
        score=score,
        score_breakdown={"total": score, "safety": 1.0},
        success=True,
        cost=0.03,
        latency_ms=900,
        token_count=320,
    )
