from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolsGenome,
)
from goagentx.core.task import Task, TaskRun
from goagentx.evolution.selection import (
    ParentSelectionSettings,
    ParentSelector,
    select_parent_pool,
)
from goagentx.registry.task_store import TaskStore


def test_default_selection_keeps_top_60_percent_and_excludes_low_scores(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    strategies = [
        _strategy("strategy-high"),
        _strategy("strategy-mid"),
        _strategy("strategy-ok"),
        _strategy("strategy-low"),
        _strategy("strategy-worst"),
    ]
    scores = {
        "strategy-high": 0.95,
        "strategy-mid": 0.88,
        "strategy-ok": 0.8,
        "strategy-low": 0.3,
        "strategy-worst": 0.2,
    }
    for index, strategy in enumerate(strategies):
        _save_run(
            store,
            run_id=f"run-{strategy.id}",
            strategy_id=strategy.id,
            task_type="doc_qa",
            score=scores[strategy.id],
            created_at=_base_time() + timedelta(minutes=index),
        )

    result = select_parent_pool(strategies, store, task_type="doc_qa")

    assert result.reason == "selected"
    assert result.insufficient_parent_pool is False
    assert result.parent_ids == ["strategy-high", "strategy-mid", "strategy-ok"]
    assert result.elite_ids == ["strategy-high"]
    assert "strategy-low" not in result.parent_ids
    assert "strategy-worst" not in result.parent_ids


def test_selection_uses_task_type_specific_history(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    strategies = [
        _strategy("doc-strong", task_type="doc_qa"),
        _strategy("doc-steady", task_type="doc_qa"),
        _strategy("global-swing", task_type=None),
        _strategy("code-specialist", task_type="code_review"),
    ]
    _save_run(
        store,
        run_id="run-doc-strong",
        strategy_id="doc-strong",
        task_type="doc_qa",
        score=0.92,
        created_at=_base_time(),
    )
    _save_run(
        store,
        run_id="run-doc-steady",
        strategy_id="doc-steady",
        task_type="doc_qa",
        score=0.81,
        created_at=_base_time(),
    )
    _save_run(
        store,
        run_id="run-global-doc",
        strategy_id="global-swing",
        task_type="doc_qa",
        score=0.2,
        created_at=_base_time(),
    )
    _save_run(
        store,
        run_id="run-global-code",
        strategy_id="global-swing",
        task_type="code_review",
        score=0.99,
        created_at=_base_time(),
    )
    _save_run(
        store,
        run_id="run-code-specialist",
        strategy_id="code-specialist",
        task_type="code_review",
        score=0.99,
        created_at=_base_time(),
    )

    result = select_parent_pool(strategies, store, task_type="doc_qa")

    average_by_strategy = {
        performance.strategy.id: performance.average_score
        for performance in result.ranked_performances
    }
    assert result.parent_ids == ["doc-strong", "doc-steady"]
    assert average_by_strategy["global-swing"] == pytest.approx(0.2)
    assert "code-specialist" not in average_by_strategy


def test_selector_expands_pool_to_minimum_parent_count(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    strategies = [
        _strategy("strategy-high"),
        _strategy("strategy-mid"),
        _strategy("strategy-low"),
    ]
    for index, strategy in enumerate(strategies):
        _save_run(
            store,
            run_id=f"run-{strategy.id}",
            strategy_id=strategy.id,
            task_type="doc_qa",
            score=0.9 - index * 0.1,
            created_at=_base_time() + timedelta(minutes=index),
        )
    settings = ParentSelectionSettings(selection_ratio=0.34, min_parent_count=2)

    result = ParentSelector(store, settings).select(strategies, task_type="doc_qa")

    assert result.parent_ids == ["strategy-high", "strategy-mid"]
    assert result.insufficient_parent_pool is False


def test_selection_marks_insufficient_when_too_few_strategies_have_history(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    strategies = [_strategy("strategy-only"), _strategy("strategy-no-history")]
    _save_run(
        store,
        run_id="run-strategy-only",
        strategy_id="strategy-only",
        task_type="doc_qa",
        score=0.9,
        created_at=_base_time(),
    )

    result = select_parent_pool(strategies, store, task_type="doc_qa")

    assert result.parent_ids == ["strategy-only"]
    assert result.elite_ids == ["strategy-only"]
    assert result.insufficient_parent_pool is True
    assert result.reason == "insufficient_scored_strategies"


def test_selection_handles_empty_scored_history(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")

    result = select_parent_pool([_strategy("strategy-no-history")], store)

    assert result.parent_ids == []
    assert result.elite_ids == []
    assert result.insufficient_parent_pool is True
    assert result.reason == "no_scored_strategies"


def _strategy(
    strategy_id: str,
    *,
    task_type: str | None = "doc_qa",
) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type=task_type,
        status=StrategyStatus.CHAMPION,
        genome=_sample_genome(),
    )


def _sample_genome() -> Genome:
    return Genome(
        model=ModelGenome(
            provider="openai_compatible",
            name="gpt-4.1",
            temperature=0.4,
            top_p=0.9,
        ),
        prompt_genome=PromptGenome(
            role="senior_code_reviewer",
            reasoning_style="evidence_first",
            risk_policy="strict",
            output_format="findings_first",
        ),
        tools=ToolsGenome(enabled=["repo_search", "shell_readonly", "browser"]),
    )


def _save_run(
    store: TaskStore,
    *,
    run_id: str,
    strategy_id: str,
    task_type: str,
    score: float,
    created_at: datetime,
) -> None:
    task_id = f"task-{run_id}"
    store.save_task(
        Task(
            id=task_id,
            task_type=task_type,
            bucket="baseline",
            input_json={"question": "What is GoAgentX?"},
            expected_json={"contains": ["GoAgentX"]},
            tags=[task_type],
            created_at=created_at,
        )
    )
    store.save_run(
        TaskRun(
            id=run_id,
            task_id=task_id,
            strategy_id=strategy_id,
            output_json={"answer": "GoAgentX evolves strategies."},
            score=score,
            score_breakdown={"quality": score},
            success=True,
            cost=0.03,
            latency_ms=900,
            token_count=320,
            tool_calls=[{"name": "repo_search", "ok": True}],
            created_at=created_at,
        )
    )


def _base_time() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)
