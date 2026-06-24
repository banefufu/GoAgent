from datetime import UTC, datetime

import pytest

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    PairedTaskResult,
)
from goagentx.arena.stats import (
    PermutationAlternative,
    SignificanceTestError,
    permutation_test_paired_result,
    permutation_test_score_deltas,
)
from goagentx.core.task import TaskRun


def test_obvious_advantage_has_low_p_value() -> None:
    result = permutation_test_score_deltas([0.2] * 8)

    assert result.insufficient_sample is False
    assert result.is_significant is True
    assert result.p_value == pytest.approx(1 / 256)
    assert result.observed_mean_delta == pytest.approx(0.2)
    assert result.permutations == 256


def test_small_sample_is_marked_insufficient() -> None:
    result = permutation_test_score_deltas([0.4, 0.5, 0.6, 0.7])

    assert result.insufficient_sample is True
    assert result.is_significant is False
    assert result.p_value is None
    assert result.reason == "need at least 5 non-zero paired deltas, got 4"


def test_balanced_sample_does_not_look_significant() -> None:
    result = permutation_test_score_deltas(
        [0.2, -0.2, 0.1, -0.1, 0.05, -0.05],
        alternative=PermutationAlternative.GREATER,
    )

    assert result.insufficient_sample is False
    assert result.observed_mean_delta == pytest.approx(0.0)
    assert result.p_value is not None
    assert result.p_value > 0.05
    assert result.is_significant is False


def test_sampled_permutation_is_stable_with_seed() -> None:
    deltas = [0.2, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.2]

    first = permutation_test_score_deltas(
        deltas,
        exact_max_samples=4,
        permutations=200,
        seed=123,
    )
    second = permutation_test_score_deltas(
        deltas,
        exact_max_samples=4,
        permutations=200,
        seed=123,
    )

    assert first == second
    assert first.permutations == 200
    assert first.seed == 123


def test_two_sided_alternative_handles_negative_delta() -> None:
    result = permutation_test_score_deltas(
        [-0.2] * 8,
        alternative=PermutationAlternative.TWO_SIDED,
    )

    assert result.p_value == pytest.approx(2 / 256)
    assert result.is_significant is True


def test_paired_result_helper_uses_evaluation_score_deltas() -> None:
    evaluation = PairedEvaluationResult(
        experiment_id="eval-stats",
        champion_strategy_id="champion-default",
        candidate_strategy_id="candidate-001",
        task_count=5,
        wins=5,
        losses=0,
        ties=0,
        win_rate=1.0,
        avg_score_delta=0.2,
        results=[
            _paired_task_result(task_id=f"task-{index}", score_delta=0.2)
            for index in range(5)
        ],
    )

    result = permutation_test_paired_result(evaluation)

    assert result.sample_count == 5
    assert result.effective_sample_count == 5
    assert result.p_value == pytest.approx(1 / 32)
    assert result.is_significant is True


def test_empty_deltas_raise_clear_error() -> None:
    with pytest.raises(SignificanceTestError, match="at least one score delta"):
        permutation_test_score_deltas([])


def _paired_task_result(task_id: str, score_delta: float) -> PairedTaskResult:
    champion_score = 0.5
    candidate_score = champion_score + score_delta
    return PairedTaskResult(
        task_id=task_id,
        task_type="doc_qa",
        bucket="baseline",
        champion_run=_task_run(
            run_id=f"run-champion-{task_id}",
            task_id=task_id,
            strategy_id="champion-default",
            score=champion_score,
        ),
        candidate_run=_task_run(
            run_id=f"run-candidate-{task_id}",
            task_id=task_id,
            strategy_id="candidate-001",
            score=candidate_score,
        ),
        score_delta=score_delta,
        outcome=PairOutcome.WIN,
    )


def _task_run(
    *,
    run_id: str,
    task_id: str,
    strategy_id: str,
    score: float,
) -> TaskRun:
    return TaskRun(
        id=run_id,
        task_id=task_id,
        strategy_id=strategy_id,
        output_json={"answer": "GoAgentX evolves strategies."},
        score=score,
        score_breakdown={"total": score},
        success=True,
        cost=0.03,
        latency_ms=900,
        token_count=320,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
