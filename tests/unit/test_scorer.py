import pytest

from goagentx.config.settings import load_settings
from goagentx.core.scoring import Scorer, ScoringInput
from goagentx.core.task import TaskRun


def test_scorer_uses_configured_weights_and_normalization() -> None:
    scorer = Scorer(load_settings().scoring)

    result = scorer.score(
        ScoringInput(
            quality_score=0.8,
            cost=0.25,
            latency_ms=2500,
            safety_violation_count=0,
        )
    )

    assert result.score == pytest.approx(0.81)
    assert result.breakdown["cost"] == pytest.approx(0.75)
    assert result.breakdown["latency"] == pytest.approx(0.75)
    assert result.breakdown["safety"] == 1.0


def test_scorer_is_deterministic_for_same_input() -> None:
    scorer = Scorer(load_settings().scoring)
    scoring_input = ScoringInput(
        quality_score=0.61,
        cost=0.17,
        latency_ms=1234,
        safety_violation_count=0,
    )

    first = scorer.score(scoring_input)
    second = scorer.score(scoring_input)

    assert first == second


def test_cost_and_latency_scores_floor_at_zero() -> None:
    scorer = Scorer(load_settings().scoring)

    result = scorer.score(
        ScoringInput(
            quality_score=1.0,
            cost=10.0,
            latency_ms=100000,
            safety_violation_count=0,
        )
    )

    assert result.breakdown["cost"] == 0.0
    assert result.breakdown["latency"] == 0.0


def test_safety_violation_directly_lowers_score() -> None:
    scorer = Scorer(load_settings().scoring)
    safe = scorer.score(
        ScoringInput(
            quality_score=0.9,
            cost=0.1,
            latency_ms=1000,
            safety_violation_count=0,
        )
    )
    unsafe = scorer.score(
        ScoringInput(
            quality_score=0.9,
            cost=0.1,
            latency_ms=1000,
            safety_violation_count=1,
        )
    )

    assert safe.score > 0.8
    assert unsafe.score == 0.0
    assert unsafe.breakdown["safety_penalty"] == 1.0


def test_score_task_run_populates_score_and_breakdown() -> None:
    scorer = Scorer(load_settings().scoring)
    task_run = TaskRun(
        id="run-001",
        task_id="task-001",
        strategy_id="strategy-001",
        output_json={"answer": "GoAgentX evolves strategies."},
        score=0.0,
        success=True,
        cost=0.25,
        latency_ms=2500,
        token_count=500,
    )

    scored_run = scorer.score_task_run(task_run, quality_score=0.8)

    assert scored_run.score == pytest.approx(0.81)
    assert scored_run.score_breakdown["total"] == pytest.approx(0.81)
    assert task_run.score == 0.0
