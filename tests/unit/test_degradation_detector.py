from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goagentx.config.settings import EvolutionSettings
from goagentx.core.task import Task, TaskRun
from goagentx.evolution.scheduler import DegradationDetector, detect_score_degradation
from goagentx.registry.task_store import TaskStore


def test_score_drop_of_15_percent_triggers_degradation() -> None:
    runs = _runs(
        baseline_scores=[0.8, 0.8, 0.8, 0.8],
        recent_scores=[0.68, 0.68, 0.68],
    )

    result = detect_score_degradation(
        runs,
        recent_window=3,
        baseline_window=4,
        threshold=0.15,
        strategy_id="champion-docs",
    )

    assert result.triggered is True
    assert result.reason == "score_drop_threshold_exceeded"
    assert result.recent_avg_score == pytest.approx(0.68)
    assert result.baseline_avg_score == pytest.approx(0.8)
    assert result.score_drop == pytest.approx(0.15)


def test_normal_score_fluctuation_does_not_trigger() -> None:
    runs = _runs(
        baseline_scores=[0.8, 0.81, 0.79, 0.8],
        recent_scores=[0.77, 0.78, 0.76],
    )

    result = detect_score_degradation(
        runs,
        recent_window=3,
        baseline_window=4,
        threshold=0.15,
    )

    assert result.triggered is False
    assert result.reason == "score_drop_within_threshold"
    assert result.score_drop < 0.15


def test_insufficient_history_does_not_trigger() -> None:
    runs = _runs(baseline_scores=[0.8], recent_scores=[0.7, 0.7])

    result = detect_score_degradation(
        runs,
        recent_window=3,
        baseline_window=4,
        threshold=0.15,
    )

    assert result.triggered is False
    assert result.reason == "insufficient_history"
    assert result.recent_sample_count == 3
    assert result.baseline_sample_count == 0
    assert result.score_drop is None


def test_detector_filters_history_by_task_type(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    settings = EvolutionSettings(
        degradation_window=3,
        baseline_window=4,
        degradation_threshold=0.15,
    )
    detector = DegradationDetector(store, settings)
    _save_strategy_history(
        store,
        strategy_id="champion-default",
        task_type="doc_qa",
        baseline_scores=[0.8, 0.8, 0.8, 0.8],
        recent_scores=[0.68, 0.68, 0.68],
    )
    _save_strategy_history(
        store,
        strategy_id="champion-default",
        task_type="code_review",
        baseline_scores=[0.8, 0.8, 0.8, 0.8],
        recent_scores=[0.78, 0.78, 0.78],
    )

    doc_result = detector.detect(strategy_id="champion-default", task_type="doc_qa")
    code_result = detector.detect(
        strategy_id="champion-default",
        task_type="code_review",
    )

    assert doc_result.triggered is True
    assert doc_result.task_type == "doc_qa"
    assert code_result.triggered is False
    assert code_result.task_type == "code_review"


def _runs(
    *,
    baseline_scores: list[float],
    recent_scores: list[float],
    strategy_id: str = "champion-docs",
) -> list[TaskRun]:
    """Build newest-first TaskRuns for detector tests."""
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    chronological_scores = baseline_scores + recent_scores
    chronological_runs = [
        _task_run(
            run_id=f"run-{index:03d}",
            task_id=f"task-{index:03d}",
            strategy_id=strategy_id,
            score=score,
            created_at=base_time + timedelta(minutes=index),
        )
        for index, score in enumerate(chronological_scores)
    ]
    return list(reversed(chronological_runs))


def _save_strategy_history(
    store: TaskStore,
    *,
    strategy_id: str,
    task_type: str,
    baseline_scores: list[float],
    recent_scores: list[float],
) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    chronological_scores = baseline_scores + recent_scores
    for index, score in enumerate(chronological_scores):
        task_id = f"{task_type}-{index:03d}"
        store.save_task(
            Task(
                id=task_id,
                task_type=task_type,
                bucket="baseline",
                input_json={"question": "What is GoAgentX?"},
                expected_json={"contains": ["GoAgentX"]},
                tags=[task_type],
                created_at=base_time,
            )
        )
        store.save_run(
            _task_run(
                run_id=f"run-{task_id}",
                task_id=task_id,
                strategy_id=strategy_id,
                score=score,
                created_at=base_time + timedelta(minutes=index),
            )
        )


def _task_run(
    *,
    run_id: str,
    task_id: str,
    strategy_id: str,
    score: float,
    created_at: datetime,
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
        created_at=created_at,
    )
