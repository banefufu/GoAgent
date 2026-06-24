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
from goagentx.evolution.mutation import (
    MutationError,
    MutationKind,
    MutationSettings,
    StrategyMutator,
    load_mutation_settings,
)


def test_load_mutation_settings_from_default_config() -> None:
    settings = load_mutation_settings()

    assert settings.parameters.temperature.max == 1.2
    assert "repo_search" in settings.tools.allowlist


def test_parameter_mutation_stays_within_config_and_records_parent() -> None:
    strategy = _strategy()
    settings = load_mutation_settings()

    candidate = StrategyMutator(settings, seed=1).mutate(
        strategy,
        kind=MutationKind.PARAMETER,
        candidate_id="candidate-parameter",
    )

    assert candidate.id == "candidate-parameter"
    assert candidate.status is StrategyStatus.CANDIDATE
    assert candidate.parent_ids == [strategy.id]
    assert candidate.version == strategy.version + 1
    assert 0.0 <= candidate.genome.model.temperature <= 1.2
    assert 0.5 <= candidate.genome.model.top_p <= 1.0
    assert 4 <= candidate.genome.tool_policy.max_calls <= 20
    assert 0 <= candidate.genome.retry_policy.max_retries <= 4
    assert candidate.genome != strategy.genome


def test_prompt_mutation_changes_whole_prompt_module() -> None:
    strategy = _strategy()
    settings = load_mutation_settings()

    candidate = StrategyMutator(settings, seed=0).mutate(
        strategy,
        kind="prompt",
        candidate_id="candidate-prompt",
    )

    parent_prompt = strategy.genome.prompt_genome.model_dump()
    candidate_prompt = candidate.genome.prompt_genome.model_dump()
    changed_modules = [
        key for key, value in candidate_prompt.items() if value != parent_prompt[key]
    ]

    assert candidate.status is StrategyStatus.CANDIDATE
    assert candidate.parent_ids == [strategy.id]
    assert len(changed_modules) == 1
    changed_module = changed_modules[0]
    assert candidate_prompt[changed_module] in getattr(settings.prompt, changed_module)


def test_tool_mutation_respects_allowlist_and_drops_disallowed_tools() -> None:
    strategy = _strategy(tools=["repo_search", "unsafe_shell"])
    settings = load_mutation_settings()

    candidate = StrategyMutator(settings, seed=4).mutate(
        strategy,
        kind=MutationKind.TOOL,
        candidate_id="candidate-tool",
    )

    assert candidate.parent_ids == [strategy.id]
    assert set(candidate.genome.tools.enabled) <= set(settings.tools.allowlist)
    assert "unsafe_shell" not in candidate.genome.tools.enabled
    assert settings.tools.min_enabled <= len(candidate.genome.tools.enabled)
    assert len(candidate.genome.tools.enabled) <= settings.tools.max_enabled


def test_each_mutation_kind_returns_valid_strategy() -> None:
    strategy = _strategy()
    settings = load_mutation_settings()
    mutator = StrategyMutator(settings, seed=7)

    candidates = [
        mutator.mutate(strategy, kind=kind, candidate_id=f"candidate-{kind.value}")
        for kind in MutationKind
    ]

    assert [candidate.status for candidate in candidates] == [
        StrategyStatus.CANDIDATE,
        StrategyStatus.CANDIDATE,
        StrategyStatus.CANDIDATE,
    ]
    assert {candidate.parent_ids[0] for candidate in candidates} == {strategy.id}
    assert len({candidate.id for candidate in candidates}) == 3


def test_invalid_mutation_config_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "mutations.yaml").write_text(
        """
parameters:
  temperature: {min: 1.0, max: 0.0, delta: 0.1}
  top_p: {min: 0.5, max: 1.0, delta: 0.05}
  max_tool_calls: {min: 4, max: 20, delta: 2}
  max_retries: {min: 0, max: 4, delta: 1}
prompt:
  role: [senior_code_reviewer]
  reasoning_style: [evidence_first]
  risk_policy: [strict]
  output_format: [findings_first]
tools:
  allowlist: [repo_search]
  min_enabled: 1
  max_enabled: 1
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(MutationError):
        load_mutation_settings(config_dir)


def _strategy(
    strategy_id: str = "champion-default",
    *,
    tools: list[str] | None = None,
) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type="code_review",
        status=StrategyStatus.CHAMPION,
        genome=_sample_genome(tools=tools),
    )


def _sample_genome(*, tools: list[str] | None = None) -> Genome:
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
        tools=ToolsGenome(enabled=tools or ["repo_search", "shell_readonly", "browser"]),
    )
