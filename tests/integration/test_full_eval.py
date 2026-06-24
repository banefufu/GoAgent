from dataclasses import dataclass
from pathlib import Path

from goagentx.arena.report import FullEvalVerdict, run_full_eval_from_settings
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
from goagentx.registry.experiment_store import EvalExperimentStore
from goagentx.registry.task_store import TaskStore


def test_full_eval_writes_report_and_eval_experiment(tmp_path: Path) -> None:
    tasks = _tasks()
    settings = load_settings()
    scorer = Scorer(settings.scoring)
    database_path = tmp_path / "goagentx.db"
    task_store = TaskStore(database_path)
    experiment_store = EvalExperimentStore(database_path)

    result = run_full_eval_from_settings(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-strong", StrategyStatus.CANDIDATE),
        tasks=tasks,
        champion_runner=QualityRunner({task.id: 0.5 for task in tasks}),
        candidate_runner=QualityRunner({task.id: 0.9 for task in tasks}),
        scorer=scorer,
        arena_settings=settings.arena,
        gate_settings=settings.promotion_gate,
        task_set_id="full-eval-fixture",
        experiment_id="full-eval-strong",
        report_directory=tmp_path / "reports",
        experiment_store=experiment_store,
        task_store=task_store,
    )

    stored_experiment = experiment_store.get("full-eval-strong")
    stored_runs = task_store.list_recent_runs(
        limit=20,
        experiment_id="full-eval-strong",
    )
    report_text = result.report_path.read_text(encoding="utf-8")

    assert result.verdict is FullEvalVerdict.PROMOTE_READY
    assert result.full_eval_passed is True
    assert result.report_path == tmp_path / "reports" / "full-eval-strong.md"
    assert result.report_path.exists()
    assert stored_experiment.report_path == result.report_path
    assert stored_experiment.verdict == "promote_ready"
    assert len(stored_runs) == len(result.selected_task_ids) * 2
    assert "# Arena Full Eval Report" in report_text
    assert "`promote_ready`" in report_text
    assert "passed_all_full_eval_checks" in report_text
    assert "Bucket Results" in report_text


def test_full_eval_report_explains_rejected_candidate(tmp_path: Path) -> None:
    tasks = _tasks()
    settings = load_settings()
    scorer = Scorer(settings.scoring)

    result = run_full_eval_from_settings(
        champion=_strategy("champion-default", StrategyStatus.CHAMPION),
        candidate=_strategy("candidate-weak", StrategyStatus.CANDIDATE),
        tasks=tasks,
        champion_runner=QualityRunner({task.id: 0.9 for task in tasks}),
        candidate_runner=QualityRunner({task.id: 0.2 for task in tasks}),
        scorer=scorer,
        arena_settings=settings.arena,
        gate_settings=settings.promotion_gate,
        task_set_id="full-eval-fixture",
        experiment_id="full-eval-weak",
        report_directory=tmp_path / "reports",
    )

    report_text = result.report_path.read_text(encoding="utf-8")

    assert result.verdict is FullEvalVerdict.REJECT
    assert result.full_eval_passed is False
    assert "win_rate_below_threshold" in result.failed_checks
    assert "avg_score_delta_below_threshold" in result.failed_checks
    assert "critical_bucket_regression" in result.failed_checks
    assert "`reject`" in report_text
    assert "win_rate_below_threshold" in report_text
    assert "Failure Examples" in report_text


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
