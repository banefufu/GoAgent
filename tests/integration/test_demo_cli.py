from pathlib import Path

from typer.testing import CliRunner

from goagentx.cli import app
from goagentx.core.strategy import StrategyStatus
from goagentx.promotion.controller import PromotionController
from goagentx.promotion.gate import (
    PromotionDecision,
    PromotionGateMetrics,
    PromotionGateResult,
)
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore


runner = CliRunner()


def test_demo_seed_creates_champion_candidate_and_tasks(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"

    first = runner.invoke(
        app,
        [
            "demo",
            "seed",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
        ],
    )
    second = runner.invoke(
        app,
        [
            "demo",
            "seed",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
        ],
    )

    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    PromotionController(registry).promote(
        "candidate_good",
        target_status=StrategyStatus.SHADOW,
        gate=_approved_gate(),
        reason="demo_test_promote",
    )
    third = runner.invoke(
        app,
        [
            "demo",
            "seed",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
        ],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert third.exit_code == 0, third.output
    assert "Seeded champion: champion (champion)" in first.output
    assert "Seeded candidate: candidate_good (candidate)" in first.output
    assert "Seeded task set: golden-agent-tasks (6 tasks)" in first.output
    assert registry.get("champion").status is StrategyStatus.CHAMPION
    assert registry.get("candidate_good").status is StrategyStatus.SHADOW
    assert len(task_store.list_tasks(task_set_id="golden-agent-tasks")) == 6


def _approved_gate() -> PromotionGateResult:
    return PromotionGateResult(
        decision=PromotionDecision.APPROVE,
        approved=True,
        failed_checks=[],
        metrics=PromotionGateMetrics(
            experiment_id="demo-test-eval",
            champion_id="champion",
            candidate_id="candidate_good",
            win_rate=1.0,
            p_value=0.01,
            avg_score_delta=0.2,
            cost_delta=0.0,
            latency_delta=0.0,
            safety_violation_count=0,
            critical_bucket_regression=False,
        ),
    )
