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
from goagentx.registry.strategy_io import (
    StrategyIOError,
    export_strategy_yaml,
    import_strategy_yaml,
    load_strategy_yaml,
)
from goagentx.registry.strategy_registry import StrategyRegistry


def test_strategy_yaml_round_trip_preserves_content(tmp_path: Path) -> None:
    strategy = _strategy("candidate-001", StrategyStatus.CANDIDATE)
    path = tmp_path / "candidate.yaml"

    export_strategy_yaml(strategy, path)
    loaded = load_strategy_yaml(path)

    assert loaded == strategy


def test_load_strategy_yaml_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
id: invalid
version: 1
name: Invalid
status: live
genome: {}
parent_ids: []
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(StrategyIOError) as exc_info:
        load_strategy_yaml(path)

    assert "Invalid strategy YAML schema" in str(exc_info.value)
    assert "status" in str(exc_info.value)


def test_import_strategy_yaml_creates_candidate_in_registry(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    source_path = tmp_path / "champion.yaml"
    export_strategy_yaml(_strategy("import-001", StrategyStatus.CHAMPION), source_path)

    imported = import_strategy_yaml(
        registry,
        source_path,
        status=StrategyStatus.CANDIDATE,
    )

    assert imported.status is StrategyStatus.CANDIDATE
    assert registry.get(imported.id).status is StrategyStatus.CANDIDATE


def test_import_strategy_yaml_allows_draft_status(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    source_path = tmp_path / "draft.yaml"
    export_strategy_yaml(_strategy("import-001", StrategyStatus.CHAMPION), source_path)

    imported = import_strategy_yaml(registry, source_path, status=StrategyStatus.DRAFT)

    assert imported.status is StrategyStatus.DRAFT


def test_import_strategy_yaml_rejects_non_importable_status(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    source_path = tmp_path / "champion.yaml"
    export_strategy_yaml(_strategy("import-001", StrategyStatus.CHAMPION), source_path)

    with pytest.raises(StrategyIOError) as exc_info:
        import_strategy_yaml(registry, source_path, status=StrategyStatus.CHAMPION)

    assert "draft or candidate" in str(exc_info.value)


def test_example_champion_yaml_is_valid() -> None:
    loaded = load_strategy_yaml(Path("strategies/champion.yaml"))

    assert loaded.id == "champion-default"
    assert loaded.status is StrategyStatus.CHAMPION
    assert loaded.genome.prompt_genome.reasoning_style == "evidence_first"


def _strategy(strategy_id: str, status: StrategyStatus) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type="code_review",
        status=status,
        genome=_sample_genome(),
        notes="Round-trip test strategy.",
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
