"""Strategy genome domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    """Base model that rejects unknown strategy fields."""

    model_config = ConfigDict(extra="forbid")


class StrategyStatus(StrEnum):
    """Allowed lifecycle statuses for a strategy."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    CANARY = "canary"
    CHAMPION = "champion"
    RETIRED = "retired"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


def _utc_now() -> datetime:
    """Return the current UTC time with timezone information."""
    return datetime.now(UTC)


class ModelGenome(StrictModel):
    """LLM model selection and sampling parameters."""

    provider: NonEmptyString
    name: NonEmptyString
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class PromptGenome(StrictModel):
    """Module-level prompt dimensions used to compose a strategy prompt."""

    role: NonEmptyString
    reasoning_style: NonEmptyString
    risk_policy: NonEmptyString
    output_format: NonEmptyString


class ToolsGenome(StrictModel):
    """Enabled tool identifiers for the strategy."""

    enabled: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("enabled")
    @classmethod
    def ensure_unique_tools(cls, value: list[str]) -> list[str]:
        """Reject duplicate tool identifiers."""
        if len(value) != len(set(value)):
            raise ValueError("enabled tools must be unique")
        return value


class ToolPolicy(StrictModel):
    """Runtime limits and tool-use preferences."""

    max_calls: int = Field(default=12, gt=0)
    prefer_read_before_edit: bool = True


class RetryPolicy(StrictModel):
    """Retry behavior for recoverable strategy execution failures."""

    max_retries: int = Field(default=2, ge=0)
    retry_on_tool_error: bool = True


class MemoryPolicy(StrictModel):
    """Project and long-term memory access policy."""

    read_project_memory: bool = True
    write_long_term_memory: Literal["never", "guarded", "always"] = "guarded"


class Genome(StrictModel):
    """Complete strategy genome."""

    model: ModelGenome
    prompt_genome: PromptGenome
    tools: ToolsGenome = Field(default_factory=ToolsGenome)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)


class Strategy(StrictModel):
    """Versioned strategy with lifecycle status and parent lineage."""

    id: NonEmptyString
    version: int = Field(..., gt=0)
    name: NonEmptyString
    status: StrategyStatus
    genome: Genome
    parent_ids: list[NonEmptyString] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    notes: str | None = None

    @field_validator("parent_ids")
    @classmethod
    def ensure_unique_parent_ids(cls, value: list[str]) -> list[str]:
        """Reject duplicate parent identifiers."""
        if len(value) != len(set(value)):
            raise ValueError("parent_ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_timestamps_and_lineage(self) -> "Strategy":
        """Validate update time and prevent self-parenting."""
        if self.id in self.parent_ids:
            raise ValueError("strategy cannot list itself as a parent")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        return self
