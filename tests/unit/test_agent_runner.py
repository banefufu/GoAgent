from pathlib import Path

import pytest

from goagentx.adapters.agent_runner import FakeAgentRunner, StaticQualityRunner
from goagentx.config.settings import load_settings
from goagentx.core.run import make_task_run_id, run_agent_task
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolsGenome,
)
from goagentx.core.task import Task, load_task_set


FIXTURE_PATH = Path("tests/fixtures/task_sets/sample_task_set.json")


def test_fake_agent_runner_returns_stable_doc_fixture_result() -> None:
    strategy = _strategy("champion-docs", task_type="doc_qa")
    task = _fixture_task("task-doc-001")
    runner = FakeAgentRunner()

    first = runner.run(strategy, task)
    second = runner.run(strategy, task)

    assert first == second
    assert first.success is True
    assert first.quality_score == 1.0
    assert "structured agent strategies" in first.output_json["answer"]


def test_fake_agent_runner_returns_code_review_finding() -> None:
    strategy = _strategy("champion-code", task_type="code_review")
    task = _fixture_task("task-code-001")

    result = FakeAgentRunner().run(strategy, task)

    assert result.quality_score == 1.0
    assert result.output_json["findings"][0]["message"] == (
        "subtraction used instead of addition"
    )


def test_run_agent_task_returns_scored_task_run() -> None:
    strategy = _strategy("champion-docs", task_type="doc_qa")
    task = _fixture_task("task-doc-001")
    scorer = Scorer(load_settings().scoring)

    task_run = run_agent_task(
        strategy=strategy,
        task=task,
        runner=FakeAgentRunner(),
        scorer=scorer,
        experiment_id="exp-c4",
    )

    assert task_run.id == "run-exp-c4-champion-docs-task-doc-001"
    assert task_run.task_id == task.id
    assert task_run.strategy_id == strategy.id
    assert task_run.experiment_id == "exp-c4"
    assert task_run.success is True
    assert task_run.score == pytest.approx(0.988)
    assert task_run.score_breakdown["quality"] == 1.0
    assert task_run.tool_calls == [
        {"name": "fake_agent", "ok": True, "task_id": "task-doc-001"}
    ]


def test_failed_fake_run_returns_low_scored_task_run() -> None:
    strategy = _strategy("champion-docs", task_type="doc_qa")
    task = _fixture_task("task-doc-001")
    scorer = Scorer(load_settings().scoring)

    task_run = run_agent_task(
        strategy=strategy,
        task=task,
        runner=FakeAgentRunner(failed_task_ids=frozenset({task.id})),
        scorer=scorer,
        run_id="run-failed",
    )

    assert task_run.id == "run-failed"
    assert task_run.success is False
    assert task_run.error_json is not None
    assert task_run.score == pytest.approx(0.288)
    assert task_run.score_breakdown["quality"] == 0.0


def test_static_quality_runner_uses_strategy_quality() -> None:
    task = _fixture_task("task-doc-001")

    result = StaticQualityRunner({"candidate-docs": 0.93}).run(
        _strategy("candidate-docs", task_type="doc_qa"),
        task,
    )

    assert result.quality_score == 0.93
    assert result.output_json["strategy_id"] == "candidate-docs"
    assert result.tool_calls[0]["name"] == "static_quality_runner"


def test_make_task_run_id_is_stable_with_and_without_experiment() -> None:
    assert make_task_run_id(strategy_id="s1", task_id="t1") == "run-s1-t1"
    assert (
        make_task_run_id(strategy_id="s1", task_id="t1", experiment_id="exp1")
        == "run-exp1-s1-t1"
    )


def _fixture_task(task_id: str) -> Task:
    task_set = load_task_set(FIXTURE_PATH)
    for task in task_set.tasks:
        if task.id == task_id:
            return task
    raise AssertionError(f"Missing fixture task: {task_id}")


def _strategy(strategy_id: str, *, task_type: str) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type=task_type,
        status=StrategyStatus.CHAMPION,
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
