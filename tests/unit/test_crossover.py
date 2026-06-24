import pytest

from goagentx.core.strategy import (
    Genome,
    MemoryPolicy,
    ModelGenome,
    PromptGenome,
    RetryPolicy,
    Strategy,
    StrategyStatus,
    ToolPolicy,
    ToolsGenome,
)
from goagentx.evolution.crossover import (
    CrossoverError,
    CrossoverSettings,
    StrategyCrossover,
    prompt_module_crossover,
    uniform_crossover,
)
from goagentx.evolution.mutation import load_mutation_settings


def test_uniform_crossover_inherits_numeric_fields_and_tracks_two_parents() -> None:
    parent_a = _strategy(
        "parent-a",
        version=2,
        temperature=0.2,
        top_p=0.7,
        max_calls=8,
        max_retries=1,
        tools=["repo_search", "unsafe_shell", "browser"],
    )
    parent_b = _strategy(
        "parent-b",
        version=5,
        temperature=0.9,
        top_p=1.0,
        max_calls=16,
        max_retries=4,
        tools=["browser", "python_readonly", "unsafe_admin"],
    )

    candidate = uniform_crossover(
        parent_a,
        parent_b,
        settings=_settings(),
        seed=3,
        candidate_id="candidate-uniform",
    )

    assert candidate.id == "candidate-uniform"
    assert candidate.status is StrategyStatus.CANDIDATE
    assert candidate.parent_ids == ["parent-a", "parent-b"]
    assert candidate.version == 6
    assert candidate.task_type == "code_review"
    assert candidate.genome.model.temperature in {0.2, 0.9}
    assert candidate.genome.model.top_p in {0.7, 1.0}
    assert candidate.genome.tool_policy.max_calls in {8, 16}
    assert candidate.genome.retry_policy.max_retries in {1, 4}
    assert _inherits_visible_values_from_both_parents(
        candidate.genome,
        parent_a.genome,
        parent_b.genome,
    )


def test_uniform_crossover_filters_tools_through_allowlist() -> None:
    parent_a = _strategy(
        "parent-a",
        tools=["repo_search", "unsafe_shell", "browser"],
    )
    parent_b = _strategy(
        "parent-b",
        tools=["browser", "python_readonly", "unsafe_admin"],
    )

    candidate = StrategyCrossover(_settings(), seed=6).uniform_crossover(
        parent_a,
        parent_b,
        candidate_id="candidate-tools",
    )

    assert set(candidate.genome.tools.enabled) <= {
        "repo_search",
        "browser",
        "python_readonly",
    }
    assert "unsafe_shell" not in candidate.genome.tools.enabled
    assert "unsafe_admin" not in candidate.genome.tools.enabled
    assert 1 <= len(candidate.genome.tools.enabled) <= 2


def test_prompt_module_crossover_inherits_prompt_modules_from_both_parents() -> None:
    parent_a = _strategy(
        "parent-a",
        role="senior_code_reviewer",
        reasoning_style="evidence_first",
        risk_policy="strict",
        output_format="findings_first",
        tools=["repo_search", "unsafe_shell"],
    )
    parent_b = _strategy(
        "parent-b",
        role="cautious_debugger",
        reasoning_style="hypothesis_driven",
        risk_policy="conservative",
        output_format="risk_summary",
        tools=["browser", "python_readonly"],
    )

    candidate = prompt_module_crossover(
        parent_a,
        parent_b,
        settings=_settings(),
        seed=9,
        candidate_id="candidate-prompt",
    )

    prompt = candidate.genome.prompt_genome
    parent_a_prompt = parent_a.genome.prompt_genome
    parent_b_prompt = parent_b.genome.prompt_genome
    assert candidate.status is StrategyStatus.CANDIDATE
    assert candidate.parent_ids == ["parent-a", "parent-b"]
    assert prompt.role in {parent_a_prompt.role, parent_b_prompt.role}
    assert prompt.reasoning_style in {
        parent_a_prompt.reasoning_style,
        parent_b_prompt.reasoning_style,
    }
    assert prompt.risk_policy in {
        parent_a_prompt.risk_policy,
        parent_b_prompt.risk_policy,
    }
    assert prompt.output_format in {
        parent_a_prompt.output_format,
        parent_b_prompt.output_format,
    }
    assert _prompt_inherits_from_both_parents(prompt, parent_a_prompt, parent_b_prompt)
    assert candidate.genome.model == parent_a.genome.model
    assert candidate.genome.tools.enabled == ["repo_search"]


def test_crossover_can_reuse_mutation_allowlist() -> None:
    settings = load_mutation_settings()
    parent_a = _strategy("parent-a", tools=["repo_search", "unsafe_shell"])
    parent_b = _strategy("parent-b", tools=["python_readonly", "unsafe_admin"])

    candidate = StrategyCrossover(settings, seed=4).uniform_crossover(
        parent_a,
        parent_b,
        candidate_id="candidate-config",
    )

    assert set(candidate.genome.tools.enabled) <= set(settings.tools.allowlist)
    assert "unsafe_shell" not in candidate.genome.tools.enabled
    assert "unsafe_admin" not in candidate.genome.tools.enabled


def test_crossover_rejects_incompatible_task_types() -> None:
    parent_a = _strategy("parent-a", task_type="doc_qa")
    parent_b = _strategy("parent-b", task_type="code_review")

    with pytest.raises(CrossoverError, match="task_type"):
        uniform_crossover(parent_a, parent_b, settings=_settings())


def test_crossover_rejects_same_parent_id() -> None:
    parent = _strategy("parent-a")

    with pytest.raises(CrossoverError, match="distinct"):
        uniform_crossover(parent, parent, settings=_settings())


def _settings() -> CrossoverSettings:
    return CrossoverSettings(
        tool_allowlist=["repo_search", "browser", "python_readonly"],
        min_enabled_tools=1,
        max_enabled_tools=2,
    )


def _strategy(
    strategy_id: str,
    *,
    version: int = 1,
    task_type: str | None = "code_review",
    provider: str = "openai_compatible",
    model_name: str = "gpt-4.1",
    temperature: float = 0.4,
    top_p: float = 0.9,
    role: str = "senior_code_reviewer",
    reasoning_style: str = "evidence_first",
    risk_policy: str = "strict",
    output_format: str = "findings_first",
    tools: list[str] | None = None,
    max_calls: int = 12,
    max_retries: int = 2,
) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=version,
        name=strategy_id,
        task_type=task_type,
        status=StrategyStatus.CHAMPION,
        genome=Genome(
            model=ModelGenome(
                provider=provider,
                name=model_name,
                temperature=temperature,
                top_p=top_p,
            ),
            prompt_genome=PromptGenome(
                role=role,
                reasoning_style=reasoning_style,
                risk_policy=risk_policy,
                output_format=output_format,
            ),
            tools=ToolsGenome(
                enabled=tools or ["repo_search", "shell_readonly", "browser"]
            ),
            tool_policy=ToolPolicy(max_calls=max_calls),
            retry_policy=RetryPolicy(max_retries=max_retries),
            memory_policy=MemoryPolicy(),
        ),
    )


def _inherits_visible_values_from_both_parents(
    child: Genome,
    parent_a: Genome,
    parent_b: Genome,
) -> bool:
    comparable_values = [
        (child.model.temperature, parent_a.model.temperature, parent_b.model.temperature),
        (child.model.top_p, parent_a.model.top_p, parent_b.model.top_p),
        (
            child.tool_policy.max_calls,
            parent_a.tool_policy.max_calls,
            parent_b.tool_policy.max_calls,
        ),
        (
            child.retry_policy.max_retries,
            parent_a.retry_policy.max_retries,
            parent_b.retry_policy.max_retries,
        ),
    ]
    inherited_from_a = any(
        child_value == parent_a_value and child_value != parent_b_value
        for child_value, parent_a_value, parent_b_value in comparable_values
    )
    inherited_from_b = any(
        child_value == parent_b_value and child_value != parent_a_value
        for child_value, parent_a_value, parent_b_value in comparable_values
    )
    return inherited_from_a and inherited_from_b


def _prompt_inherits_from_both_parents(
    prompt: PromptGenome,
    parent_a: PromptGenome,
    parent_b: PromptGenome,
) -> bool:
    comparable_values = [
        (prompt.role, parent_a.role, parent_b.role),
        (prompt.reasoning_style, parent_a.reasoning_style, parent_b.reasoning_style),
        (prompt.risk_policy, parent_a.risk_policy, parent_b.risk_policy),
        (prompt.output_format, parent_a.output_format, parent_b.output_format),
    ]
    inherited_from_a = any(
        child_value == parent_a_value and child_value != parent_b_value
        for child_value, parent_a_value, parent_b_value in comparable_values
    )
    inherited_from_b = any(
        child_value == parent_b_value and child_value != parent_a_value
        for child_value, parent_a_value, parent_b_value in comparable_values
    )
    return inherited_from_a and inherited_from_b
