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
from goagentx.core.task import load_task_set
from goagentx.registry.experiment_store import EvalExperimentStore
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore


runner = CliRunner()


def test_eval_cli_runs_full_eval_from_task_set_file(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    report_dir = tmp_path / "reports"
    registry = StrategyRegistry(database_path)
    registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    registry.create(_strategy("candidate-docs", StrategyStatus.CANDIDATE))

    result = runner.invoke(
        app,
        [
            "eval",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--champion",
            "champion-docs",
            "--candidate",
            "candidate-docs",
            "--task-set",
            "tests/fixtures/task_sets/sample_task_set.json",
            "--report-dir",
            str(report_dir),
            "--experiment-id",
            "cli-full-eval-file",
            "--seed",
            "3",
        ],
    )

    experiment = EvalExperimentStore(database_path).get("cli-full-eval-file")
    stored_runs = TaskStore(database_path).list_recent_runs(
        limit=20,
        experiment_id="cli-full-eval-file",
    )
    report_path = report_dir / "cli-full-eval-file.md"

    assert result.exit_code == 0, result.output
    assert "Full Eval verdict: reject" in result.output
    assert "Experiment: cli-full-eval-file" in result.output
    assert "Report:" in result.output
    assert "Failed checks:" in result.output
    assert report_path.exists()
    assert experiment.report_path == report_path
    assert experiment.task_set_id == "sample-agent-tasks"
    assert len(stored_runs) == 4


def test_eval_cli_runs_full_eval_from_stored_task_set_id(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    registry.create(_strategy("candidate-docs", StrategyStatus.CANDIDATE))
    task_store.save_task_set(load_task_set("tests/fixtures/task_sets/sample_task_set.json"))

    result = runner.invoke(
        app,
        [
            "eval",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--champion",
            "champion-docs",
            "--candidate",
            "candidate-docs",
            "--task-set",
            "sample-agent-tasks",
            "--report-dir",
            str(tmp_path / "reports"),
            "--experiment-id",
            "cli-full-eval-db",
        ],
    )

    experiment = EvalExperimentStore(database_path).get("cli-full-eval-db")

    assert result.exit_code == 0, result.output
    assert "Task set: sample-agent-tasks" in result.output
    assert "Selected tasks: 2" in result.output
    assert experiment.id == "cli-full-eval-db"


def test_eval_cli_exits_when_task_set_is_missing(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    registry.create(_strategy("candidate-docs", StrategyStatus.CANDIDATE))

    result = runner.invoke(
        app,
        [
            "eval",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--champion",
            "champion-docs",
            "--candidate",
            "candidate-docs",
            "--task-set",
            "missing-task-set",
        ],
    )

    assert result.exit_code == 1
    assert "Task set not found: missing-task-set" in result.output


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
