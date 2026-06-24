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
from goagentx.registry.strategy_io import export_strategy_yaml, load_strategy_yaml
from goagentx.registry.strategy_registry import StrategyRegistry


runner = CliRunner()


def test_strategy_cli_lists_and_shows_registered_strategies(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    champion = registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    candidate = registry.create(_strategy("candidate-docs", StrategyStatus.CANDIDATE))

    list_result = runner.invoke(
        app,
        [
            "strategy",
            "list",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--status",
            "candidate",
            "--task-type",
            "doc_qa",
        ],
    )
    show_result = runner.invoke(
        app,
        [
            "strategy",
            "show",
            champion.id,
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
        ],
    )

    assert list_result.exit_code == 0, list_result.output
    assert "id\tstatus\tversion\ttask_type\tname" in list_result.output
    assert candidate.id in list_result.output
    assert champion.id not in list_result.output
    assert show_result.exit_code == 0, show_result.output
    assert "id: champion-docs" in show_result.output
    assert "status: champion" in show_result.output
    assert "task_type: doc_qa" in show_result.output


def test_strategy_cli_imports_and_exports_yaml(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    source_path = tmp_path / "source.yaml"
    exported_path = tmp_path / "exported" / "candidate.yaml"
    export_strategy_yaml(_strategy("imported-docs", StrategyStatus.CHAMPION), source_path)

    import_result = runner.invoke(
        app,
        [
            "strategy",
            "import",
            str(source_path),
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--status",
            "draft",
        ],
    )
    export_result = runner.invoke(
        app,
        [
            "strategy",
            "export",
            "imported-docs",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--output",
            str(exported_path),
        ],
    )

    loaded = load_strategy_yaml(exported_path)

    assert import_result.exit_code == 0, import_result.output
    assert "Imported strategy imported-docs as draft." in import_result.output
    assert StrategyRegistry(database_path).get("imported-docs").status is StrategyStatus.DRAFT
    assert export_result.exit_code == 0, export_result.output
    assert "Exported strategy imported-docs" in export_result.output
    assert loaded.id == "imported-docs"
    assert loaded.status is StrategyStatus.DRAFT


def test_strategy_cli_rejects_non_importable_status(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    source_path = tmp_path / "source.yaml"
    export_strategy_yaml(_strategy("imported-docs", StrategyStatus.CHAMPION), source_path)

    result = runner.invoke(
        app,
        [
            "strategy",
            "import",
            str(source_path),
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--status",
            "champion",
        ],
    )

    assert result.exit_code == 1
    assert "draft or candidate" in result.output


def test_strategy_cli_show_missing_strategy_exits_with_error(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    StrategyRegistry(database_path)

    result = runner.invoke(
        app,
        [
            "strategy",
            "show",
            "missing",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
        ],
    )

    assert result.exit_code == 1
    assert "Strategy not found: missing" in result.output


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
