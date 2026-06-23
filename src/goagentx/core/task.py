"""Task and task-run domain models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

NonEmptyString = Annotated[str, Field(min_length=1)]


class TaskModelError(RuntimeError):
    """Raised when a task-set fixture cannot be loaded."""


class StrictModel(BaseModel):
    """Base model that rejects unknown task fields."""

    model_config = ConfigDict(extra="forbid")


def _utc_now() -> datetime:
    """Return the current UTC time with timezone information."""
    return datetime.now(UTC)


class Task(StrictModel):
    """A reusable task that can be replayed across strategies."""

    id: NonEmptyString
    task_type: NonEmptyString
    bucket: NonEmptyString
    input_json: dict[str, Any]
    expected_json: dict[str, Any] | None = None
    tags: list[NonEmptyString] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("input_json", "expected_json")
    @classmethod
    def ensure_json_mapping(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject values that cannot be serialized to JSON."""
        _assert_json_serializable(value)
        return value

    @field_validator("tags")
    @classmethod
    def ensure_unique_tags(cls, value: list[str]) -> list[str]:
        """Reject duplicate tag labels."""
        if len(value) != len(set(value)):
            raise ValueError("tags must be unique")
        return value


class TaskSet(StrictModel):
    """A named collection of tasks for repeatable evaluation."""

    id: NonEmptyString
    tasks: list[Task] = Field(..., min_length=1)
    description: str | None = None


class TaskRun(StrictModel):
    """The result of executing one strategy against one task."""

    id: NonEmptyString
    task_id: NonEmptyString
    strategy_id: NonEmptyString
    experiment_id: NonEmptyString | None = None
    output_json: dict[str, Any]
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    success: bool
    cost: float = Field(..., ge=0.0)
    latency_ms: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error_json: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("output_json", "error_json")
    @classmethod
    def ensure_json_output(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject output/error payloads that cannot be serialized to JSON."""
        _assert_json_serializable(value)
        return value

    @field_validator("tool_calls")
    @classmethod
    def ensure_tool_calls_are_json(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reject tool-call payloads that cannot be serialized to JSON."""
        _assert_json_serializable(value)
        return value


def load_task_set(path: str | Path) -> TaskSet:
    """Load a TaskSet fixture from a JSON file."""
    source_path = Path(path)
    try:
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskModelError(f"Task set fixture not found: {source_path}") from exc
    except OSError as exc:
        raise TaskModelError(f"Failed to read task set fixture: {source_path}") from exc
    except json.JSONDecodeError as exc:
        raise TaskModelError(f"Invalid task set JSON: {source_path}") from exc

    try:
        return TaskSet.model_validate(loaded)
    except ValueError as exc:
        raise TaskModelError(f"Invalid task set schema: {source_path}\n{exc}") from exc

def _assert_json_serializable(value: Any) -> None:
    """Raise ValueError if a value cannot be serialized as JSON."""
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError("value must be JSON serializable") from exc
