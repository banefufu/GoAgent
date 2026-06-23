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
from goagentx.registry.strategy_registry import (
    StrategyAlreadyExistsError,
    StrategyNotFoundError,
    StrategyRegistry,
)


def test_create_and_get_strategy_round_trips_model(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    strategy = _strategy("candidate-001", StrategyStatus.CANDIDATE)

    created = registry.create(strategy)
    loaded = registry.get(strategy.id)

    assert created == loaded
    assert loaded.genome.prompt_genome.role == "senior_code_reviewer"
    assert loaded.parent_ids == []
    assert loaded.status is StrategyStatus.CANDIDATE


def test_duplicate_strategy_id_is_rejected(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    strategy = _strategy("candidate-001", StrategyStatus.CANDIDATE)
    registry.create(strategy)

    with pytest.raises(StrategyAlreadyExistsError):
        registry.create(strategy)


def test_list_by_status_returns_matching_strategies(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))
    registry.create(_strategy("draft-001", StrategyStatus.DRAFT))

    candidates = registry.list_by_status("candidate")

    assert candidates == [candidate]


def test_get_champion_returns_task_type_champion(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    champion = registry.create(
        _strategy("champion-code-001", StrategyStatus.CHAMPION, task_type="code_review")
    )
    registry.create(
        _strategy("champion-docs-001", StrategyStatus.CHAMPION, task_type="docs")
    )

    loaded = registry.get_champion("code_review")

    assert loaded.id == champion.id
    assert loaded.task_type == "code_review"


def test_create_new_champion_retires_existing_same_task_type(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    old_champion = registry.create(
        _strategy("champion-code-001", StrategyStatus.CHAMPION, task_type="code_review")
    )
    new_champion = registry.create(
        _strategy("champion-code-002", StrategyStatus.CHAMPION, task_type="code_review")
    )

    assert registry.get_champion("code_review").id == new_champion.id
    assert registry.get(old_champion.id).status is StrategyStatus.RETIRED


def test_global_champion_is_separate_from_task_type_champion(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    global_champion = registry.create(_strategy("champion-global", StrategyStatus.CHAMPION))
    task_champion = registry.create(
        _strategy("champion-code", StrategyStatus.CHAMPION, task_type="code_review")
    )

    assert registry.get_champion().id == global_champion.id
    assert registry.get_champion("code_review").id == task_champion.id


def test_update_status_refreshes_updated_at(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))

    registry.update_status(candidate.id, StrategyStatus.REJECTED)

    loaded = registry.get(candidate.id)
    assert loaded.status is StrategyStatus.REJECTED
    assert loaded.updated_at > candidate.updated_at


def test_update_status_to_champion_retires_existing_champion(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    old_champion = registry.create(
        _strategy("champion-code-001", StrategyStatus.CHAMPION, task_type="code_review")
    )
    candidate = registry.create(
        _strategy("candidate-code-001", StrategyStatus.CANDIDATE, task_type="code_review")
    )

    registry.update_status(candidate.id, StrategyStatus.CHAMPION)

    assert registry.get(candidate.id).status is StrategyStatus.CHAMPION
    assert registry.get(old_champion.id).status is StrategyStatus.RETIRED
    assert registry.get_champion("code_review").id == candidate.id


def test_get_missing_strategy_raises(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")

    with pytest.raises(StrategyNotFoundError):
        registry.get("missing")


def _strategy(
    strategy_id: str,
    status: StrategyStatus,
    *,
    task_type: str | None = None,
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
