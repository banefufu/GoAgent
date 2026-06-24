from dataclasses import dataclass

import pytest

from goagentx.arena.runner import (
    QuickRejectDecision,
    QuickRejectError,
    run_quick_reject_from_settings,
    select_quick_reject_tasks,
)
from goagentx.config.settings import load_settings
from goagentx.core.run import AgentRunResult
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolsGenome,
)
from goagentx.core.task import Task


def test_quick_reject_rejects_weak_candidate_before_full_eval() -> None:
    tasks = _tasks()
    settings = load_settings()
    scorer = Scorer(settings.scoring)

    result = run_quick_reject_from_settings(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-weak", StrategyStatus.CANDIDATE),
        tasks=tasks,
        champion_runner=QualityRunner({task.id: 0.9 for task in tasks}),
        candidate_runner=QualityRunner({task.id: 0.2 for task in tasks}),
        scorer=scorer,
        settings=settings.arena,
        experiment_id="quick-reject-weak",
    )

    assert result.decision is QuickRejectDecision.REJECT
    assert result.quick_reject_passed is False
    assert result.evaluation.task_count == settings.arena.quick_reject_rounds
    assert result.evaluation.win_rate == 0.0
    assert result.evaluation.avg_score_delta < 0.0
    assert "win_rate_below_threshold" in result.failed_checks
    assert "avg_score_delta_below_threshold" in result.failed_checks
    assert "significant_regression" in result.failed_checks
    assert result.regression_significance.is_significant is True


def test_quick_reject_passes_strong_candidate_to_full_eval() -> None:
    tasks = _tasks()
    settings = load_settings()
    scorer = Scorer(settings.scoring)

    result = run_quick_reject_from_settings(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-strong", StrategyStatus.CANDIDATE),
        tasks=tasks,
        champion_runner=QualityRunner({task.id: 0.5 for task in tasks}),
        candidate_runner=QualityRunner({task.id: 0.9 for task in tasks}),
        scorer=scorer,
        settings=settings.arena,
        experiment_id="quick-reject-strong",
    )

    assert result.decision is QuickRejectDecision.PASS_TO_FULL_EVAL
    assert result.quick_reject_passed is True
    assert result.reason == "passed_quick_reject"
    assert result.failed_checks == []
    assert result.evaluation.win_rate == 1.0
    assert result.evaluation.avg_score_delta > 0.0


def test_quick_reject_sampling_is_bucket_stratified() -> None:
    selected_tasks = select_quick_reject_tasks(_tasks(), rounds=3, seed=7)

    assert len(selected_tasks) == 3
    assert {task.bucket for task in selected_tasks} == {
        "baseline",
        "critical",
        "edge",
    }


def test_quick_reject_rejects_empty_task_set() -> None:
    with pytest.raises(QuickRejectError, match="at least one task"):
        select_quick_reject_tasks([], rounds=5)


@dataclass(frozen=True)
class QualityRunner:
    quality_by_task_id: dict[str, float]
    cost: float = 0.03
    latency_ms: int = 900
    token_count: int = 320

    def run(self, strategy: Strategy, task: Task) -> AgentRunResult:
        quality = self.quality_by_task_id[task.id]
        return AgentRunResult(
            output_json={
                "strategy_id": strategy.id,
                "task_id": task.id,
                "quality": quality,
            },
            quality_score=quality,
            success=True,
            cost=self.cost,
            latency_ms=self.latency_ms,
            token_count=self.token_count,
            tool_calls=[
                {
                    "name": "quality_runner",
                    "strategy_id": strategy.id,
                    "task_id": task.id,
                }
            ],
        )


def _tasks() -> list[Task]:
    return [
        _task("task-baseline-001", bucket="baseline"),
        _task("task-baseline-002", bucket="baseline"),
        _task("task-critical-001", bucket="critical"),
        _task("task-critical-002", bucket="critical"),
        _task("task-edge-001", bucket="edge"),
        _task("task-edge-002", bucket="edge"),
    ]


def _task(task_id: str, *, bucket: str) -> Task:
    return Task(
        id=task_id,
        task_type="doc_qa",
        bucket=bucket,
        input_json={"question": "What is GoAgentX?"},
        expected_json={"contains": ["GoAgentX"]},
        tags=["doc_qa", bucket],
    )


def _strategy(strategy_id: str, status: StrategyStatus) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type="doc_qa",
        status=status,
        genome=_sample_genome(),
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
