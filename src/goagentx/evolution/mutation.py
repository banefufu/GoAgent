"""Strategy mutation utilities for DreamCycle."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from goagentx.config.settings import DEFAULT_CONFIG_DIR
from goagentx.core.strategy import Genome, Strategy, StrategyStatus


MUTATION_CONFIG_FILE = "mutations.yaml"


class MutationError(RuntimeError):
    """Raised when mutation configuration or execution fails."""


class MutationKind(StrEnum):
    """Supported mutation categories."""

    PARAMETER = "parameter"
    PROMPT = "prompt"
    TOOL = "tool"


class StrictModel(BaseModel):
    """Base model that rejects unknown mutation fields."""

    model_config = ConfigDict(extra="forbid")


class NumericMutationRange(StrictModel):
    """Allowed numeric mutation range for one parameter."""

    min: float
    max: float
    delta: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NumericMutationRange":
        """Ensure numeric bounds are usable."""
        if self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class ParameterMutationSettings(StrictModel):
    """Configurable ranges for parameter mutations."""

    temperature: NumericMutationRange
    top_p: NumericMutationRange
    max_tool_calls: NumericMutationRange
    max_retries: NumericMutationRange


class PromptMutationSettings(StrictModel):
    """Allowed module-level prompt substitutions."""

    role: list[str] = Field(..., min_length=1)
    reasoning_style: list[str] = Field(..., min_length=1)
    risk_policy: list[str] = Field(..., min_length=1)
    output_format: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_options(self) -> "PromptMutationSettings":
        """Reject duplicate options in each prompt module."""
        for field_name in ("role", "reasoning_style", "risk_policy", "output_format"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} options must be unique")
        return self


class ToolMutationSettings(StrictModel):
    """Tool mutation allowlist and size bounds."""

    allowlist: list[str] = Field(..., min_length=1)
    min_enabled: int = Field(..., ge=0)
    max_enabled: int = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_allowlist_and_bounds(self) -> "ToolMutationSettings":
        """Ensure tool allowlist and bounds are consistent."""
        if len(self.allowlist) != len(set(self.allowlist)):
            raise ValueError("tool allowlist must be unique")
        if self.min_enabled > self.max_enabled:
            raise ValueError("min_enabled cannot be greater than max_enabled")
        if self.max_enabled > len(self.allowlist):
            raise ValueError("max_enabled cannot exceed allowlist size")
        return self


class MutationSettings(StrictModel):
    """Complete mutation configuration."""

    parameters: ParameterMutationSettings
    prompt: PromptMutationSettings
    tools: ToolMutationSettings


class StrategyMutator:
    """Generate candidate strategies by mutating one genome dimension."""

    def __init__(self, settings: MutationSettings, *, seed: int | None = 0) -> None:
        """Create a deterministic mutator when seed is provided."""
        self.settings = settings
        self._rng = random.Random(seed)

    def mutate(
        self,
        strategy: Strategy,
        *,
        kind: MutationKind | str,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Return a legal candidate strategy from a single mutation kind."""
        resolved_kind = MutationKind(kind)
        if resolved_kind is MutationKind.PARAMETER:
            return self.mutate_parameters(strategy, candidate_id=candidate_id)
        if resolved_kind is MutationKind.PROMPT:
            return self.mutate_prompt(strategy, candidate_id=candidate_id)
        return self.mutate_tools(strategy, candidate_id=candidate_id)

    def mutate_parameters(
        self,
        strategy: Strategy,
        *,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Mutate one numeric model/tool/retry parameter."""
        genome_data = strategy.genome.model_dump(mode="python")
        field_name = self._rng.choice(
            ["temperature", "top_p", "max_tool_calls", "max_retries"]
        )
        if field_name in {"temperature", "top_p"}:
            model_data = genome_data["model"]
            range_config = getattr(self.settings.parameters, field_name)
            model_data[field_name] = _mutate_number(
                float(model_data[field_name]),
                range_config,
                rng=self._rng,
            )
        elif field_name == "max_tool_calls":
            tool_policy_data = genome_data["tool_policy"]
            tool_policy_data["max_calls"] = int(
                _mutate_number(
                    float(tool_policy_data["max_calls"]),
                    self.settings.parameters.max_tool_calls,
                    rng=self._rng,
                    integer=True,
                )
            )
        else:
            retry_policy_data = genome_data["retry_policy"]
            retry_policy_data["max_retries"] = int(
                _mutate_number(
                    float(retry_policy_data["max_retries"]),
                    self.settings.parameters.max_retries,
                    rng=self._rng,
                    integer=True,
                )
            )

        return _candidate_from_genome(
            strategy,
            genome=Genome.model_validate(genome_data),
            kind=MutationKind.PARAMETER,
            candidate_id=candidate_id,
        )

    def mutate_prompt(
        self,
        strategy: Strategy,
        *,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Mutate one prompt module as a whole value."""
        genome_data = strategy.genome.model_dump(mode="python")
        prompt_data = genome_data["prompt_genome"]
        module_name = self._rng.choice(
            ["role", "reasoning_style", "risk_policy", "output_format"]
        )
        prompt_data[module_name] = _choose_replacement(
            current=prompt_data[module_name],
            options=getattr(self.settings.prompt, module_name),
            rng=self._rng,
        )

        return _candidate_from_genome(
            strategy,
            genome=Genome.model_validate(genome_data),
            kind=MutationKind.PROMPT,
            candidate_id=candidate_id,
        )

    def mutate_tools(
        self,
        strategy: Strategy,
        *,
        candidate_id: str | None = None,
    ) -> Strategy:
        """Mutate enabled tools while respecting the safety allowlist."""
        genome_data = strategy.genome.model_dump(mode="python")
        allowlist = self.settings.tools.allowlist
        enabled = [
            tool for tool in genome_data["tools"]["enabled"] if tool in set(allowlist)
        ]
        enabled = _unique_preserve_order(enabled)
        if not enabled:
            enabled = [allowlist[0]]

        can_add = len(enabled) < self.settings.tools.max_enabled and any(
            tool not in enabled for tool in allowlist
        )
        can_remove = len(enabled) > self.settings.tools.min_enabled
        if can_add and (not can_remove or self._rng.choice([True, False])):
            candidates = [tool for tool in allowlist if tool not in enabled]
            enabled.append(self._rng.choice(candidates))
        elif can_remove:
            enabled.pop(self._rng.randrange(len(enabled)))

        genome_data["tools"]["enabled"] = enabled[: self.settings.tools.max_enabled]
        return _candidate_from_genome(
            strategy,
            genome=Genome.model_validate(genome_data),
            kind=MutationKind.TOOL,
            candidate_id=candidate_id,
        )


def load_mutation_settings(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> MutationSettings:
    """Load mutation settings from configs/mutations.yaml."""
    path = Path(config_dir) / MUTATION_CONFIG_FILE
    if not path.exists():
        raise MutationError(f"Mutation config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MutationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MutationError(f"Mutation config must contain a YAML mapping: {path}")
    try:
        return MutationSettings.model_validate(data)
    except ValueError as exc:
        raise MutationError(f"Invalid mutation config:\n{exc}") from exc


def _candidate_from_genome(
    parent: Strategy,
    *,
    genome: Genome,
    kind: MutationKind,
    candidate_id: str | None,
) -> Strategy:
    """Build a candidate strategy with parent lineage recorded."""
    now = datetime.now(UTC)
    resolved_candidate_id = candidate_id or f"{parent.id}-{kind.value}-candidate"
    return Strategy(
        id=resolved_candidate_id,
        version=parent.version + 1,
        name=f"{parent.name} {kind.value} mutation",
        task_type=parent.task_type,
        status=StrategyStatus.CANDIDATE,
        genome=genome,
        parent_ids=[parent.id],
        created_at=now,
        updated_at=now,
        notes=f"Generated by {kind.value} mutation from {parent.id}.",
    )


def _mutate_number(
    current: float,
    range_config: NumericMutationRange,
    *,
    rng: random.Random,
    integer: bool = False,
) -> float:
    """Move a number by one configured delta while staying within bounds."""
    direction = rng.choice([-1.0, 1.0])
    candidate = current + direction * range_config.delta
    if candidate > range_config.max:
        candidate = current - range_config.delta
    if candidate < range_config.min:
        candidate = current + range_config.delta
    candidate = _clamp(candidate, minimum=range_config.min, maximum=range_config.max)
    if integer:
        return float(round(candidate))
    return round(candidate, 6)


def _choose_replacement(
    *,
    current: str,
    options: list[str],
    rng: random.Random,
) -> str:
    """Choose an option different from current when possible."""
    replacements = [option for option in options if option != current]
    if not replacements:
        return current
    return rng.choice(replacements)


def _unique_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate values without changing first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp a number to inclusive bounds."""
    return min(maximum, max(minimum, value))
