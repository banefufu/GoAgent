"""Agent runner adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from goagentx.core.run import AgentRunResult, AgentRunner
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task


@dataclass(frozen=True)
class FakeAgentRunner:
    """Deterministic runner used to exercise the evaluation loop."""

    cost: float = 0.03
    latency_ms: int = 900
    token_count: int = 320
    failed_task_ids: frozenset[str] = field(default_factory=frozenset)

    def run(self, strategy: Strategy, task: Task) -> AgentRunResult:
        """Return a stable fixture-based result for one strategy/task pair."""
        if task.id in self.failed_task_ids:
            return AgentRunResult(
                output_json={},
                quality_score=0.0,
                success=False,
                cost=self.cost,
                latency_ms=self.latency_ms,
                token_count=self.token_count,
                tool_calls=[_fake_tool_call(task, ok=False)],
                error_json={
                    "type": "fake_agent_error",
                    "message": f"Fake runner was configured to fail task {task.id}.",
                },
            )

        output_json = _fixture_output(strategy, task)
        return AgentRunResult(
            output_json=output_json,
            quality_score=_quality_score(output_json, task.expected_json),
            success=True,
            cost=self.cost,
            latency_ms=self.latency_ms,
            token_count=self.token_count,
            tool_calls=[_fake_tool_call(task, ok=True)],
        )


def _fixture_output(strategy: Strategy, task: Task) -> dict[str, Any]:
    """Generate deterministic task output from fixture expectations."""
    expected = task.expected_json or {}
    if "contains" in expected:
        contains = expected["contains"]
        answer = " ".join(str(item) for item in contains)
        return {
            "answer": answer,
            "task_id": task.id,
            "strategy_id": strategy.id,
        }
    if "finding" in expected:
        return {
            "findings": [
                {
                    "message": str(expected["finding"]),
                    "severity": "high",
                }
            ],
            "task_id": task.id,
            "strategy_id": strategy.id,
        }
    return {
        "echo": task.input_json,
        "task_id": task.id,
        "strategy_id": strategy.id,
    }


def _quality_score(output_json: dict[str, Any], expected_json: dict[str, Any] | None) -> float:
    """Calculate simple fixture quality without model or judge calls."""
    if expected_json is None:
        return 1.0

    haystack = json.dumps(output_json, sort_keys=True)
    if "contains" in expected_json:
        expected_items = [str(item) for item in expected_json["contains"]]
        if not expected_items:
            return 1.0
        matched = sum(1 for item in expected_items if item in haystack)
        return matched / len(expected_items)
    if "finding" in expected_json:
        return 1.0 if str(expected_json["finding"]) in haystack else 0.0
    return 1.0


def _fake_tool_call(task: Task, *, ok: bool) -> dict[str, Any]:
    """Build a stable fake tool-call payload."""
    return {
        "name": "fake_agent",
        "ok": ok,
        "task_id": task.id,
    }


__all__ = [
    "AgentRunner",
    "AgentRunResult",
    "FakeAgentRunner",
]
