from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from goagentx.cli import app
from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolsGenome,
)
from goagentx.core.task import Task, TaskRun, load_task_set
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore


runner = CliRunner()


def test_cli_evolve_dream_generates_candidates_and_runs_quick_reject(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "goagentx.db"
    audit_log_path = tmp_path / "audit" / "dreamcycle.jsonl"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    champion = registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    task_store.save_task_set(load_task_set("tests/fixtures/task_sets/sample_task_set.json"))

    result = runner.invoke(
        app,
        [
            "evolve",
            "dream",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--strategy",
            champion.id,
            "--task-set",
            "sample-agent-tasks",
            "--candidate-count",
            "2",
            "--audit-log",
            str(audit_log_path),
            "--seed",
            "11",
        ],
    )

    registry_candidates = registry.list_by_status(StrategyStatus.CANDIDATE)
    stored_runs = task_store.list_recent_runs(limit=20)

    assert result.exit_code == 0, result.output
    assert "DreamCycle triggered: true" in result.output
    assert "Reason: manual_trigger" in result.output
    assert "Generated candidates: 2" in result.output
    assert "quick_reject=" in result.output
    assert len(registry_candidates) == 2
    assert all(candidate.parent_ids == [champion.id] for candidate in registry_candidates)
    assert audit_log_path.exists()
    assert len(stored_runs) == 4


def test_cli_evolve_ga_generates_candidates(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    for strategy_id, status, score in [
        ("parent-high", StrategyStatus.RETIRED, 0.95),
        ("parent-mid", StrategyStatus.SHADOW, 0.85),
        ("parent-ok", StrategyStatus.CANARY, 0.75),
    ]:
        registry.create(_strategy(strategy_id, status))
        _save_history(task_store, strategy_id=strategy_id, scores=[score, score])

    result = runner.invoke(
        app,
        [
            "evolve",
            "ga",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--task-type",
            "doc_qa",
            "--population-size",
            "4",
            "--population",
            "sample",
            "--mutation-rate",
            "0.34",
            "--seed",
            "3",
        ],
    )

    registry_candidates = registry.list_by_status(StrategyStatus.CANDIDATE)

    assert result.exit_code == 0, result.output
    assert "Generated 3 candidates" in result.output
    assert len(registry_candidates) == 3
    assert all(candidate.id.startswith("ga-sample-") for candidate in registry_candidates)


def _strategy(strategy_id: str, status: StrategyStatus) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type="doc_qa",
        status=status,
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


def _save_history(
    task_store: TaskStore,
    *,
    strategy_id: str,
    scores: list[float],
) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    for index, score in enumerate(scores):
        task_id = f"task-{strategy_id}-{index:03d}"
        created_at = base_time + timedelta(minutes=index)
        task_store.save_task(
            Task(
                id=task_id,
                task_type="doc_qa",
                bucket="baseline",
                input_json={"question": "What does GoAgentX evolve?"},
                expected_json={"contains": ["agent strategies"]},
                tags=["doc_qa"],
                created_at=created_at,
            )
        )
        task_store.save_run(
            TaskRun(
                id=f"run-{task_id}",
                task_id=task_id,
                strategy_id=strategy_id,
                output_json={"answer": "GoAgentX evolves agent strategies."},
                score=score,
                score_breakdown={"total": score},
                success=True,
                cost=0.03,
                latency_ms=900,
                token_count=320,
                created_at=created_at,
            )
        )
