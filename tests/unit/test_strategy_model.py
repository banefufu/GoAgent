import json

import pytest
from pydantic import ValidationError

from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolPolicy,
    ToolsGenome,
)


def test_valid_strategy_can_be_created_and_json_serialized() -> None:
    strategy = Strategy(
        id="strategy-001",
        version=1,
        name="Evidence-first reviewer",
        status=StrategyStatus.CANDIDATE,
        genome=_sample_genome(),
    )

    serialized = strategy.model_dump(mode="json")

    assert serialized["parent_ids"] == []
    assert serialized["status"] == "candidate"
    assert serialized["genome"]["model"]["temperature"] == 0.4
    json.dumps(serialized)


def test_parent_ids_default_to_independent_empty_lists() -> None:
    first = Strategy(
        id="strategy-001",
        version=1,
        name="First",
        status=StrategyStatus.DRAFT,
        genome=_sample_genome(),
    )
    second = Strategy(
        id="strategy-002",
        version=1,
        name="Second",
        status=StrategyStatus.DRAFT,
        genome=_sample_genome(),
    )

    first.parent_ids.append("parent-001")

    assert first.parent_ids == ["parent-001"]
    assert second.parent_ids == []


def test_invalid_status_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Strategy(
            id="strategy-001",
            version=1,
            name="Invalid status",
            status="live",
            genome=_sample_genome(),
        )

    assert "status" in str(exc_info.value)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("temperature", 2.5),
        ("top_p", 1.1),
    ],
)
def test_invalid_model_parameter_range_is_rejected(
    field_name: str,
    value: float,
) -> None:
    data = {
        "provider": "openai_compatible",
        "name": "gpt-4.1",
        "temperature": 0.4,
        "top_p": 0.9,
    }
    data[field_name] = value

    with pytest.raises(ValidationError) as exc_info:
        ModelGenome(**data)

    assert field_name in str(exc_info.value)


def test_invalid_tool_policy_range_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolPolicy(max_calls=0)

    assert "max_calls" in str(exc_info.value)


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolsGenome(enabled=["repo_search", "repo_search"])

    assert "enabled tools must be unique" in str(exc_info.value)


def test_self_parenting_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Strategy(
            id="strategy-001",
            version=1,
            name="Self parent",
            status=StrategyStatus.CANDIDATE,
            genome=_sample_genome(),
            parent_ids=["strategy-001"],
        )

    assert "cannot list itself as a parent" in str(exc_info.value)


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
