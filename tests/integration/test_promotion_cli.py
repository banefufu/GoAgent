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
from goagentx.promotion.controller import PromotionController
from goagentx.registry.strategy_registry import StrategyRegistry


runner = CliRunner()


def test_cli_promote_advances_candidate_to_shadow(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    candidate = registry.create(_strategy("candidate-docs", StrategyStatus.CANDIDATE))

    result = runner.invoke(
        app,
        [
            "promote",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--candidate",
            candidate.id,
            "--mode",
            "shadow",
            "--reason",
            "cli_manual_gate_passed",
        ],
    )

    events = PromotionController(registry).list_events(strategy_id=candidate.id)

    assert result.exit_code == 0, result.output
    assert "Promoted candidate-docs: candidate -> shadow" in result.output
    assert "Gate decision: approve" in result.output
    assert registry.get(candidate.id).status is StrategyStatus.SHADOW
    assert len(events) == 1
    assert events[0].reason == "cli_manual_gate_passed"
    assert events[0].experiment_id == "manual-promotion-candidate-docs"


def test_cli_promote_rejects_when_gate_metrics_fail(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    candidate = registry.create(_strategy("candidate-docs", StrategyStatus.CANDIDATE))

    result = runner.invoke(
        app,
        [
            "promote",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--candidate",
            candidate.id,
            "--mode",
            "shadow",
            "--win-rate",
            "0.1",
        ],
    )

    assert result.exit_code == 1
    assert "Promotion gate rejected" in result.output
    assert "win_rate_below_threshold" in result.output
    assert registry.get(candidate.id).status is StrategyStatus.CANDIDATE
    assert PromotionController(registry).list_events(strategy_id=candidate.id) == []


def test_cli_rollback_restores_previous_champion(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    previous = registry.create(_strategy("champion-previous", StrategyStatus.CHAMPION))
    current = registry.create(_strategy("champion-current", StrategyStatus.CHAMPION))

    result = runner.invoke(
        app,
        [
            "rollback",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--to",
            previous.id,
            "--reason",
            "cli_safety_regression",
        ],
    )

    events = PromotionController(registry).list_events()

    assert result.exit_code == 0, result.output
    assert "Rollback restored champion-previous as champion." in result.output
    assert "Failed strategy champion-current marked rolled_back." in result.output
    assert "Events written: 2" in result.output
    assert registry.get(previous.id).status is StrategyStatus.CHAMPION
    assert registry.get(current.id).status is StrategyStatus.ROLLED_BACK
    assert [event.reason for event in events] == [
        "cli_safety_regression",
        "cli_safety_regression",
    ]


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
