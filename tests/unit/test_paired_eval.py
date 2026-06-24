from dataclasses import dataclass
from pathlib import Path

import pytest

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvalError,
    evaluate_pair,
    make_paired_experiment_id,
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
from goagentx.core.task import Task, load_task_set
from goagentx.registry.task_store import TaskStore


FIXTURE_PATH = Path("tests/fixtures/task_sets/sample_task_set.json")


def test_evaluate_pair_outputs_expected_fixture_win_rate() -> None:
    task_set = load_task_set(FIXTURE_PATH)
    scorer = Scorer(load_settings().scoring)

    result = evaluate_pair(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-001", StrategyStatus.CANDIDATE),
        tasks=task_set.tasks,
        champion_runner=QualityRunner(
            {
                "task-doc-001": 0.5,
                "task-code-001": 0.9,
            }
        ),
        candidate_runner=QualityRunner(
            {
                "task-doc-001": 0.8,
                "task-code-001": 0.6,
            }
        ),
        scorer=scorer,
        tie_threshold=0.01,
        experiment_id="eval-fixture",
    )

    assert result.experiment_id == "eval-fixture"
    assert result.task_count == 2
    assert result.wins == 1
    assert result.losses == 1
    assert result.ties == 0
    assert result.win_rate == 0.5
    assert result.avg_score_delta == pytest.approx(0.0)
    assert [item.outcome for item in result.results] == [
        PairOutcome.WIN,
        PairOutcome.LOSS,
    ]
    assert result.results[0].score_delta == pytest.approx(0.21)


def test_tie_threshold_absorbs_small_score_delta() -> None:
    task = _task("task-close")
    scorer = Scorer(load_settings().scoring)

    result = evaluate_pair(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-close", StrategyStatus.CANDIDATE),
        tasks=[task],
        champion_runner=QualityRunner({"task-close": 0.8}),
        candidate_runner=QualityRunner({"task-close": 0.81}),
        scorer=scorer,
        tie_threshold=0.01,
    )

    assert result.results[0].score_delta == pytest.approx(0.007)
    assert result.results[0].outcome is PairOutcome.TIE
    assert result.win_rate == 0.0


def test_evaluate_pair_can_persist_both_task_runs(tmp_path: Path) -> None:
    task = _task("task-store")
    store = TaskStore(tmp_path / "goagentx.db")
    store.save_task(task)
    scorer = Scorer(load_settings().scoring)

    result = evaluate_pair(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-001", StrategyStatus.CANDIDATE),
        tasks=[task],
        champion_runner=QualityRunner({"task-store": 0.4}),
        candidate_runner=QualityRunner({"task-store": 0.9}),
        scorer=scorer,
        experiment_id="eval-store",
        task_store=store,
    )

    stored_runs = store.list_recent_runs(limit=10, experiment_id="eval-store")

    assert result.wins == 1
    assert {run.strategy_id for run in stored_runs} == {
        "champion-default",
        "candidate-001",
    }
    assert {run.task_id for run in stored_runs} == {"task-store"}


def test_evaluate_pair_rejects_empty_tasks() -> None:
    scorer = Scorer(load_settings().scoring)

    with pytest.raises(PairedEvalError, match="at least one task"):
        evaluate_pair(
            champion=_strategy("champion-default", StrategyStatus.CHAMPION),
            candidate=_strategy("candidate-001", StrategyStatus.CANDIDATE),
            tasks=[],
            champion_runner=QualityRunner({}),
            scorer=scorer,
        )


def test_make_paired_experiment_id_is_stable() -> None:
    assert (
        make_paired_experiment_id(
            champion_strategy_id="champion-default",
            candidate_strategy_id="candidate-001",
        )
        == "eval-champion-default-vs-candidate-001"
    )


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


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        task_type="doc_qa",
        bucket="baseline",
        input_json={"question": "What is GoAgentX?"},
        expected_json={"contains": ["GoAgentX"]},
        tags=["doc_qa", "baseline"],
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
