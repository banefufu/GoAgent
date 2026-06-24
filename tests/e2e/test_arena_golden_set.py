from dataclasses import dataclass
from pathlib import Path

from goagentx.arena.report import FullEvalVerdict, run_full_eval_from_settings
from goagentx.config.settings import load_settings
from goagentx.core.run import AgentRunResult
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task, TaskSet, load_task_set
from goagentx.registry.experiment_store import EvalExperimentStore
from goagentx.registry.strategy_io import load_strategy_yaml
from goagentx.registry.task_store import TaskStore


TASK_SET_PATH = Path("tests/fixtures/task_sets/sample_agent_tasks.json")
CHAMPION_PATH = Path("tests/fixtures/strategies/champion.yaml")
GOOD_CANDIDATE_PATH = Path("tests/fixtures/strategies/candidate_good.yaml")
BAD_CANDIDATE_PATH = Path("tests/fixtures/strategies/candidate_bad.yaml")


def test_arena_golden_set_accepts_good_candidate(tmp_path: Path) -> None:
    task_set = load_task_set(TASK_SET_PATH)
    champion = load_strategy_yaml(CHAMPION_PATH)
    candidate = load_strategy_yaml(GOOD_CANDIDATE_PATH)
    settings = load_settings()
    scorer = Scorer(settings.scoring)
    task_store, experiment_store = _stores(tmp_path, task_set)

    result = run_full_eval_from_settings(
        champion=champion,
        candidate=candidate,
        tasks=task_set.tasks,
        champion_runner=GoldenQualityRunner({champion.id: 0.60, candidate.id: 0.95}),
        scorer=scorer,
        arena_settings=settings.arena,
        gate_settings=settings.promotion_gate,
        task_set_id=task_set.id,
        experiment_id="golden-good-candidate",
        report_directory=tmp_path / "reports",
        experiment_store=experiment_store,
        task_store=task_store,
    )

    stored_experiment = experiment_store.get(result.experiment_id)
    stored_runs = task_store.list_recent_runs(
        limit=20,
        experiment_id=result.experiment_id,
    )
    report_text = result.report_path.read_text(encoding="utf-8")

    assert result.verdict is FullEvalVerdict.PROMOTE_READY
    assert result.full_eval_passed is True
    assert result.failed_checks == []
    assert result.evaluation.win_rate == 1.0
    assert result.significance.p_value is not None
    assert result.significance.p_value <= settings.arena.p_value_threshold
    assert set(result.selected_task_ids) == {task.id for task in task_set.tasks}
    assert stored_experiment.verdict == FullEvalVerdict.PROMOTE_READY.value
    assert len(stored_runs) == len(task_set.tasks) * 2
    assert "`promote_ready`" in report_text
    assert "passed_all_full_eval_checks" in report_text


def test_arena_golden_set_rejects_bad_candidate(tmp_path: Path) -> None:
    task_set = load_task_set(TASK_SET_PATH)
    champion = load_strategy_yaml(CHAMPION_PATH)
    candidate = load_strategy_yaml(BAD_CANDIDATE_PATH)
    settings = load_settings()
    scorer = Scorer(settings.scoring)
    task_store, experiment_store = _stores(tmp_path, task_set)

    result = run_full_eval_from_settings(
        champion=champion,
        candidate=candidate,
        tasks=task_set.tasks,
        champion_runner=GoldenQualityRunner({champion.id: 0.80, candidate.id: 0.20}),
        scorer=scorer,
        arena_settings=settings.arena,
        gate_settings=settings.promotion_gate,
        task_set_id=task_set.id,
        experiment_id="golden-bad-candidate",
        report_directory=tmp_path / "reports",
        experiment_store=experiment_store,
        task_store=task_store,
    )

    stored_experiment = experiment_store.get(result.experiment_id)
    stored_runs = task_store.list_recent_runs(
        limit=20,
        experiment_id=result.experiment_id,
    )
    report_text = result.report_path.read_text(encoding="utf-8")

    assert result.verdict is FullEvalVerdict.REJECT
    assert result.full_eval_passed is False
    assert result.evaluation.win_rate == 0.0
    assert result.evaluation.avg_score_delta < 0.0
    assert set(result.selected_task_ids) == {task.id for task in task_set.tasks}
    assert "win_rate_below_threshold" in result.failed_checks
    assert "avg_score_delta_below_threshold" in result.failed_checks
    assert "critical_bucket_regression" in result.failed_checks
    assert stored_experiment.verdict == FullEvalVerdict.REJECT.value
    assert len(stored_runs) == len(task_set.tasks) * 2
    assert "`reject`" in report_text
    assert "Failure Examples" in report_text


@dataclass(frozen=True)
class GoldenQualityRunner:
    quality_by_strategy_id: dict[str, float]
    cost: float = 0.03
    latency_ms: int = 900
    token_count: int = 320

    def run(self, strategy: Strategy, task: Task) -> AgentRunResult:
        quality = self.quality_by_strategy_id[strategy.id]
        return AgentRunResult(
            output_json={
                "strategy_id": strategy.id,
                "task_id": task.id,
                "task_type": task.task_type,
                "quality": quality,
            },
            quality_score=quality,
            success=True,
            cost=self.cost,
            latency_ms=self.latency_ms,
            token_count=self.token_count,
            tool_calls=[
                {
                    "name": "golden_quality_runner",
                    "strategy_id": strategy.id,
                    "task_id": task.id,
                }
            ],
        )


def _stores(tmp_path: Path, task_set: TaskSet) -> tuple[TaskStore, EvalExperimentStore]:
    database_path = tmp_path / "goagentx.db"
    task_store = TaskStore(database_path)
    task_store.save_task_set(task_set)
    return task_store, EvalExperimentStore(database_path)
