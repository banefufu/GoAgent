"""Runtime helpers for executing strategies against tasks."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task, TaskRun


class StrictModel(BaseModel):
    """Base model that rejects unknown run fields."""

    model_config = ConfigDict(extra="forbid")


class AgentRunResult(StrictModel):
    """Raw adapter output before it is converted into a scored TaskRun."""

    output_json: dict[str, Any]
    quality_score: float = Field(..., ge=0.0, le=1.0)
    success: bool = True
    cost: float = Field(..., ge=0.0)
    latency_ms: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error_json: dict[str, Any] | None = None
    safety_violation_count: int = Field(default=0, ge=0)

    @field_validator("output_json", "error_json")
    @classmethod
    def ensure_json_mapping(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject payloads that cannot be serialized to JSON."""
        _assert_json_serializable(value)
        return value

    @field_validator("tool_calls")
    @classmethod
    def ensure_tool_calls_are_json(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reject non-JSON tool-call payloads."""
        _assert_json_serializable(value)
        return value


class AgentRunner(Protocol):
    """Adapter interface implemented by fake and real agent runners."""

    def run(self, strategy: Strategy, task: Task) -> AgentRunResult:
        """Execute one strategy on one task and return raw run metrics."""


def run_agent_task(
    *,
    strategy: Strategy,
    task: Task,
    runner: AgentRunner,
    scorer: Scorer,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> TaskRun:
    """Execute and score one strategy/task pair as a TaskRun."""
    result = runner.run(strategy, task)
    unscored_run = TaskRun(
        id=run_id or make_task_run_id(
            strategy_id=strategy.id,
            task_id=task.id,
            experiment_id=experiment_id,
        ),
        task_id=task.id,
        strategy_id=strategy.id,
        experiment_id=experiment_id,
        output_json=result.output_json,
        score=0.0,
        success=result.success,
        cost=result.cost,
        latency_ms=result.latency_ms,
        token_count=result.token_count,
        tool_calls=result.tool_calls,
        error_json=result.error_json,
    )
    return scorer.score_task_run(
        unscored_run,
        quality_score=result.quality_score,
        safety_violation_count=result.safety_violation_count,
    )


def make_task_run_id(
    *,
    strategy_id: str,
    task_id: str,
    experiment_id: str | None = None,
) -> str:
    """Build a stable task-run id for repeatable fixture evaluations."""
    if experiment_id is None:
        return f"run-{strategy_id}-{task_id}"
    return f"run-{experiment_id}-{strategy_id}-{task_id}"


def _assert_json_serializable(value: Any) -> None:
    """Raise ValueError if a value cannot be serialized as JSON."""
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError("value must be JSON serializable") from exc
