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
from goagentx.promotion.controller import PromotionController
from goagentx.promotion.rollback import RollbackController, RollbackControllerError
from goagentx.registry.strategy_registry import StrategyRegistry


def test_rollback_restores_retired_champion_and_marks_current_rolled_back(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    previous = registry.create(
        _strategy("champion-previous", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    current = registry.create(
        _strategy("champion-current", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    controller = RollbackController(registry)

    result = controller.rollback(previous.id, reason="safety_regression")

    events = PromotionController(registry).list_events()

    assert result.restored_strategy.id == previous.id
    assert result.restored_strategy.status is StrategyStatus.CHAMPION
    assert result.failed_strategy is not None
    assert result.failed_strategy.id == current.id
    assert result.failed_strategy.status is StrategyStatus.ROLLED_BACK
    assert registry.get_champion("doc_qa").id == previous.id
    assert registry.get(current.id).status is StrategyStatus.ROLLED_BACK
    assert [event.strategy_id for event in result.events] == [
        current.id,
        previous.id,
    ]
    assert [(event.from_status, event.to_status) for event in result.events] == [
        (StrategyStatus.CHAMPION, StrategyStatus.ROLLED_BACK),
        (StrategyStatus.RETIRED, StrategyStatus.CHAMPION),
    ]
    assert [event.reason for event in result.events] == [
        "safety_regression",
        "safety_regression",
    ]
    assert events == result.events


def test_rollback_marks_canary_failed_without_changing_current_champion(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    champion = registry.create(
        _strategy("champion-stable", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    canary = registry.create(
        _strategy("candidate-canary", StrategyStatus.CANARY, task_type="doc_qa")
    )
    controller = RollbackController(registry)

    result = controller.rollback(
        champion.id,
        failed_strategy_id=canary.id,
        reason="canary_cost_spike",
    )

    events = PromotionController(registry).list_events(strategy_id=canary.id)

    assert result.restored_strategy.id == champion.id
    assert result.restored_strategy.status is StrategyStatus.CHAMPION
    assert result.failed_strategy is not None
    assert result.failed_strategy.status is StrategyStatus.ROLLED_BACK
    assert registry.get_champion("doc_qa").id == champion.id
    assert registry.get(canary.id).status is StrategyStatus.ROLLED_BACK
    assert len(result.events) == 1
    assert result.events[0].from_status is StrategyStatus.CANARY
    assert result.events[0].to_status is StrategyStatus.ROLLED_BACK
    assert result.events[0].reason == "canary_cost_spike"
    assert events == result.events


def test_rollback_can_retire_failed_strategy_when_requested(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    previous = registry.create(
        _strategy("champion-previous", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    current = registry.create(
        _strategy("champion-current", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    controller = RollbackController(registry)

    result = controller.rollback(
        previous.id,
        failed_status=StrategyStatus.RETIRED,
        reason="manual_restore",
    )

    assert result.restored_strategy.status is StrategyStatus.CHAMPION
    assert result.failed_strategy is not None
    assert result.failed_strategy.id == current.id
    assert result.failed_strategy.status is StrategyStatus.RETIRED
    assert result.events[0].to_status is StrategyStatus.RETIRED


def test_rollback_rejects_target_that_was_not_stable_champion(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    registry.create(_strategy("champion-stable", StrategyStatus.CHAMPION))
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))
    controller = RollbackController(registry)

    with pytest.raises(RollbackControllerError, match="stable champion"):
        controller.rollback(candidate.id)

    assert registry.get(candidate.id).status is StrategyStatus.CANDIDATE
    assert PromotionController(registry).list_events() == []


def test_rollback_rejects_already_champion_without_failed_strategy(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    champion = registry.create(_strategy("champion-stable", StrategyStatus.CHAMPION))
    controller = RollbackController(registry)

    with pytest.raises(RollbackControllerError, match="already champion"):
        controller.rollback(champion.id)

    assert registry.get(champion.id).status is StrategyStatus.CHAMPION
    assert PromotionController(registry).list_events() == []


def test_rollback_rejects_failed_strategy_from_different_domain(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    champion = registry.create(
        _strategy("champion-docs", StrategyStatus.CHAMPION, task_type="docs")
    )
    canary = registry.create(
        _strategy("candidate-code-canary", StrategyStatus.CANARY, task_type="code")
    )
    controller = RollbackController(registry)

    with pytest.raises(RollbackControllerError, match="same task_type"):
        controller.rollback(champion.id, failed_strategy_id=canary.id)

    assert registry.get(champion.id).status is StrategyStatus.CHAMPION
    assert registry.get(canary.id).status is StrategyStatus.CANARY
    assert PromotionController(registry).list_events() == []


def test_rollback_to_retired_target_requires_current_champion_as_failed_strategy(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    previous = registry.create(
        _strategy("champion-previous", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    current = registry.create(
        _strategy("champion-current", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    canary = registry.create(
        _strategy("candidate-canary", StrategyStatus.CANARY, task_type="doc_qa")
    )
    controller = RollbackController(registry)

    with pytest.raises(RollbackControllerError, match="current champion"):
        controller.rollback(previous.id, failed_strategy_id=canary.id)

    assert registry.get(previous.id).status is StrategyStatus.RETIRED
    assert registry.get(current.id).status is StrategyStatus.CHAMPION
    assert registry.get(canary.id).status is StrategyStatus.CANARY
    assert PromotionController(registry).list_events() == []


def _strategy(
    strategy_id: str,
    status: StrategyStatus,
    *,
    task_type: str | None = "doc_qa",
) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type=task_type,
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
