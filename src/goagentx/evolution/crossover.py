"""Strategy crossover utilities for genome GA."""

from __future__ import annotations

import random
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from goagentx.core.strategy import Genome, Strategy, StrategyStatus
from goagentx.evolution.mutation import MutationSettings, load_mutation_settings


class CrossoverError(RuntimeError):
    """Raised when two strategies cannot be crossed safely."""


class CrossoverKind(StrEnum):
    """Supported crossover categories."""

    UNIFORM = "uniform"
    PROMPT_MODULE = "prompt_module"


class StrictModel(BaseModel):
    """Base model that rejects unknown crossover fields."""

    model_config = ConfigDict(extra="forbid")


class CrossoverSettings(StrictModel):
    """Tool constraints used while crossing strategy genomes."""

    tool_allowlist: list[str] = Field(..., min_length=1)
    min_enabled_tools: int = Field(default=1, ge=0)
    max_enabled_tools: int = Field(default=4, gt=0)

    @classmethod
    def from_mutation_settings(
        cls,
        mutation_settings: MutationSettings,
    ) -> "CrossoverSettings":
        """Reuse the mutation safety allowlist for crossover."""
        return cls(
            tool_allowlist=mutation_settings.tools.allowlist,
            min_enabled_tools=mutation_settings.tools.min_enabled,
            max_enabled_tools=mutation_settings.tools.max_enabled,
        )

    @model_validator(mode="after")
    def validate_tool_bounds(self) -> "CrossoverSettings":
        """Ensure allowlist and tool-count bounds are usable."""
        if len(self.tool_allowlist) != len(set(self.tool_allowlist)):
            raise ValueError("tool_allowlist must be unique")
        if self.min_enabled_tools > self.max_enabled_tools:
            raise ValueError("min_enabled_tools cannot exceed max_enabled_tools")
        if self.max_enabled_tools > len(self.tool_allowlist):
            raise ValueError("max_enabled_tools cannot exceed tool_allowlist size")
        return self


class StrategyCrossover:
    """Generate candidate strategies by crossing two parent genomes."""

    def __init__(
        self,
        settings: CrossoverSettings | MutationSettings | None = None,
        *,
        seed: int | None = 0,
    ) -> None:
        """Create a deterministic crossover operator when seed is provided."""
        self.settings = _resolve_settings(settings)
        self._rng = random.Random(seed)

    def crossover(
        self,
        parent_a: Strategy,
        parent_b: Strategy,
        *,
        kind: CrossoverKind | str = CrossoverKind.UNIFORM,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Return a legal candidate strategy from two parents."""
        resolved_kind = CrossoverKind(kind)
        if resolved_kind is CrossoverKind.UNIFORM:
            return self.uniform_crossover(
                parent_a,
                parent_b,
                candidate_id=candidate_id,
            )
        return self.prompt_module_crossover(
            parent_a,
            parent_b,
            candidate_id=candidate_id,
        )

    def uniform_crossover(
        self,
        parent_a: Strategy,
        parent_b: Strategy,
        *,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Cross scalar genome fields independently and tools as a set."""
        _validate_parent_pair(parent_a, parent_b)
        genome_a = parent_a.genome.model_dump(mode="python")
        genome_b = parent_b.genome.model_dump(mode="python")
        child_genome = deepcopy(genome_a)
        _inherit_fields(
            child_genome,
            genome_a,
            genome_b,
            paths=[
                ("model", "provider"),
                ("model", "name"),
                ("model", "temperature"),
                ("model", "top_p"),
                ("prompt_genome", "role"),
                ("prompt_genome", "reasoning_style"),
                ("prompt_genome", "risk_policy"),
                ("prompt_genome", "output_format"),
                ("tool_policy", "max_calls"),
                ("tool_policy", "prefer_read_before_edit"),
                ("retry_policy", "max_retries"),
                ("retry_policy", "retry_on_tool_error"),
                ("memory_policy", "read_project_memory"),
                ("memory_policy", "write_long_term_memory"),
            ],
            rng=self._rng,
        )
        child_genome["tools"]["enabled"] = _crossover_tools(
            genome_a["tools"]["enabled"],
            genome_b["tools"]["enabled"],
            settings=self.settings,
            rng=self._rng,
        )

        return _candidate_from_genome(
            parent_a,
            parent_b,
            genome=Genome.model_validate(child_genome),
            kind=CrossoverKind.UNIFORM,
            candidate_id=candidate_id,
        )

    def prompt_module_crossover(
        self,
        parent_a: Strategy,
        parent_b: Strategy,
        *,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Cross prompt modules while keeping other dimensions from parent A."""
        _validate_parent_pair(parent_a, parent_b)
        genome_a = parent_a.genome.model_dump(mode="python")
        genome_b = parent_b.genome.model_dump(mode="python")
        child_genome = deepcopy(genome_a)
        _inherit_fields(
            child_genome,
            genome_a,
            genome_b,
            paths=[
                ("prompt_genome", "role"),
                ("prompt_genome", "reasoning_style"),
                ("prompt_genome", "risk_policy"),
                ("prompt_genome", "output_format"),
            ],
            rng=self._rng,
        )
        child_genome["tools"]["enabled"] = _sanitize_tools(
            genome_a["tools"]["enabled"],
            settings=self.settings,
        )

        return _candidate_from_genome(
            parent_a,
            parent_b,
            genome=Genome.model_validate(child_genome),
            kind=CrossoverKind.PROMPT_MODULE,
            candidate_id=candidate_id,
        )


def uniform_crossover(
    parent_a: Strategy,
    parent_b: Strategy,
    *,
    settings: CrossoverSettings | MutationSettings | None = None,
    seed: int | None = 0,
    candidate_id: str | None = None,
) -> Strategy:
    """Cross two strategies with uniform field inheritance."""
    return StrategyCrossover(settings, seed=seed).uniform_crossover(
        parent_a,
        parent_b,
        candidate_id=candidate_id,
    )


def prompt_module_crossover(
    parent_a: Strategy,
    parent_b: Strategy,
    *,
    settings: CrossoverSettings | MutationSettings | None = None,
    seed: int | None = 0,
    candidate_id: str | None = None,
) -> Strategy:
    """Cross two strategies by inheriting prompt modules from both parents."""
    return StrategyCrossover(settings, seed=seed).prompt_module_crossover(
        parent_a,
        parent_b,
        candidate_id=candidate_id,
    )


def _resolve_settings(
    settings: CrossoverSettings | MutationSettings | None,
) -> CrossoverSettings:
    if settings is None:
        return CrossoverSettings.from_mutation_settings(load_mutation_settings())
    if isinstance(settings, MutationSettings):
        return CrossoverSettings.from_mutation_settings(settings)
    return settings


def _validate_parent_pair(parent_a: Strategy, parent_b: Strategy) -> None:
    if parent_a.id == parent_b.id:
        raise CrossoverError("parent strategies must be distinct")
    if (
        parent_a.task_type is not None
        and parent_b.task_type is not None
        and parent_a.task_type != parent_b.task_type
    ):
        raise CrossoverError("parent task_type values are incompatible")


def _inherit_fields(
    child_genome: dict[str, Any],
    genome_a: dict[str, Any],
    genome_b: dict[str, Any],
    *,
    paths: list[tuple[str, str]],
    rng: random.Random,
) -> None:
    inherited_from_a = 0
    inherited_from_b = 0
    decisions: list[tuple[tuple[str, str], str]] = []

    for path in paths:
        source = "a" if rng.choice([True, False]) else "b"
        decisions.append((path, source))
        if source == "a":
            inherited_from_a += 1
        else:
            inherited_from_b += 1

    if inherited_from_a == 0 or inherited_from_b == 0:
        _force_mixed_inheritance(decisions, genome_a, genome_b)

    for (section, field_name), source in decisions:
        child_genome[section][field_name] = (
            genome_a if source == "a" else genome_b
        )[section][field_name]


def _force_mixed_inheritance(
    decisions: list[tuple[tuple[str, str], str]],
    genome_a: dict[str, Any],
    genome_b: dict[str, Any],
) -> None:
    if not decisions:
        return
    current_source = decisions[0][1]
    replacement_source = "b" if current_source == "a" else "a"
    for index, (path, source) in enumerate(decisions):
        section, field_name = path
        if genome_a[section][field_name] != genome_b[section][field_name]:
            decisions[index] = (path, replacement_source)
            return
    decisions[-1] = (decisions[-1][0], replacement_source)


def _crossover_tools(
    tools_a: list[str],
    tools_b: list[str],
    *,
    settings: CrossoverSettings,
    rng: random.Random,
) -> list[str]:
    allowed_tools = set(settings.tool_allowlist)
    parent_a_tools = _unique_preserve_order(
        tool for tool in tools_a if tool in allowed_tools
    )
    parent_b_tools = _unique_preserve_order(
        tool for tool in tools_b if tool in allowed_tools
    )
    inherited: list[str] = []
    for tool in settings.tool_allowlist:
        in_a = tool in parent_a_tools
        in_b = tool in parent_b_tools
        if in_a and in_b:
            inherited.append(tool)
        elif in_a or in_b:
            if rng.choice([True, False]):
                inherited.append(tool)

    if len(inherited) < settings.min_enabled_tools:
        for tool in _unique_preserve_order([*parent_a_tools, *parent_b_tools]):
            if tool not in inherited:
                inherited.append(tool)
            if len(inherited) >= settings.min_enabled_tools:
                break

    if len(inherited) < settings.min_enabled_tools:
        for tool in settings.tool_allowlist:
            if tool not in inherited:
                inherited.append(tool)
            if len(inherited) >= settings.min_enabled_tools:
                break

    if len(inherited) > settings.max_enabled_tools:
        inherited = rng.sample(inherited, settings.max_enabled_tools)
        inherited.sort(key=settings.tool_allowlist.index)

    return inherited


def _sanitize_tools(
    tools: list[str],
    *,
    settings: CrossoverSettings,
) -> list[str]:
    enabled = _unique_preserve_order(
        tool for tool in tools if tool in set(settings.tool_allowlist)
    )
    if len(enabled) < settings.min_enabled_tools:
        for tool in settings.tool_allowlist:
            if tool not in enabled:
                enabled.append(tool)
            if len(enabled) >= settings.min_enabled_tools:
                break
    return enabled[: settings.max_enabled_tools]


def _candidate_from_genome(
    parent_a: Strategy,
    parent_b: Strategy,
    *,
    genome: Genome,
    kind: CrossoverKind,
    candidate_id: str | None,
) -> Strategy:
    now = datetime.now(UTC)
    resolved_candidate_id = (
        candidate_id or f"{parent_a.id}-{parent_b.id}-{kind.value}-candidate"
    )
    return Strategy(
        id=resolved_candidate_id,
        version=max(parent_a.version, parent_b.version) + 1,
        name=f"{parent_a.name} x {parent_b.name} {kind.value} crossover",
        task_type=_resolve_child_task_type(parent_a, parent_b),
        status=StrategyStatus.CANDIDATE,
        genome=genome,
        parent_ids=[parent_a.id, parent_b.id],
        created_at=now,
        updated_at=now,
        notes=(
            f"Generated by {kind.value} crossover from "
            f"{parent_a.id} and {parent_b.id}."
        ),
    )


def _resolve_child_task_type(parent_a: Strategy, parent_b: Strategy) -> str | None:
    if parent_a.task_type == parent_b.task_type:
        return parent_a.task_type
    return parent_a.task_type or parent_b.task_type


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
