import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goagentx.arena.report import (
    FullEvaluationResult,
    FullEvalVerdict,
    run_full_eval_from_settings,
)
from goagentx.config.settings import EvolutionSettings, Settings, load_settings
from goagentx.core.run import AgentRunResult
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.core.task import Task, TaskRun, TaskSet, load_task_set
from goagentx.evolution.dreamcycle import DreamCycleResult, run_dreamcycle
from goagentx.evolution.mutation import (
    MutationKind,
    StrategyMutator,
    load_mutation_settings,
)
from goagentx.promotion.controller import PromotionController, PromotionControllerError
from goagentx.promotion.gate import evaluate_promotion_gate
from goagentx.registry.experiment_store import EvalExperimentStore
from goagentx.registry.strategy_io import load_strategy_yaml
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore


TASK_SET_PATH = Path("tests/fixtures/task_sets/sample_agent_tasks.json")
CHAMPION_PATH = Path("tests/fixtures/strategies/champion.yaml")


def test_evolution_flow_promotes_good_candidate_and_rejects_bad_candidate(
    tmp_path: Path,
) -> None:
    settings = load_settings()
    scorer = Scorer(settings.scoring)
    registry, task_store, experiment_store, task_set = _stores(tmp_path)
    champion = registry.create(load_strategy_yaml(CHAMPION_PATH))
    _save_degraded_history(task_store, champion_id=champion.id, tasks=task_set.tasks)

    dreamcycle = run_dreamcycle(
        champion_id=champion.id,
        registry=registry,
        task_store=task_store,
        mutator=StrategyMutator(load_mutation_settings(), seed=17),
        scorer=scorer,
        runner=StrategyQualityRunner({champion.id: 0.60}),
        candidate_runner=StrategyQualityRunner({}, default_quality=0.90),
        evolution_settings=_evolution_settings(),
        arena_settings=settings.arena,
        audit_log_path=tmp_path / "audit" / "dreamcycle.jsonl",
        candidate_count=2,
        mutation_kinds=[MutationKind.PARAMETER, MutationKind.PROMPT],
        auto_run_arena=True,
        seed=23,
    )
    good_candidate = dreamcycle.candidates[0].candidate
    bad_candidate = dreamcycle.candidates[1].candidate

    good_eval = _run_full_eval(
        champion=champion,
        candidate=good_candidate,
        task_set=task_set,
        scorer=scorer,
        settings=settings,
        quality_by_strategy_id={champion.id: 0.60, good_candidate.id: 0.95},
        experiment_id="i2-good-candidate",
        tmp_path=tmp_path,
        experiment_store=experiment_store,
        task_store=task_store,
    )
    good_gate = evaluate_promotion_gate(good_eval, settings.promotion_gate)
    controller = PromotionController(registry)
    promotion = controller.promote(
        good_candidate.id,
        target_status=StrategyStatus.SHADOW,
        gate=good_gate,
        reason="i2_full_eval_passed",
    )

    bad_eval = _run_full_eval(
        champion=champion,
        candidate=bad_candidate,
        task_set=task_set,
        scorer=scorer,
        settings=settings,
        quality_by_strategy_id={champion.id: 0.80, bad_candidate.id: 0.20},
        experiment_id="i2-bad-candidate",
        tmp_path=tmp_path,
        experiment_store=experiment_store,
        task_store=task_store,
    )
    bad_gate = evaluate_promotion_gate(bad_eval, settings.promotion_gate)
    with pytest.raises(PromotionControllerError, match="promotion gate rejected"):
        controller.promote(
            bad_candidate.id,
            target_status=StrategyStatus.SHADOW,
            gate=bad_gate,
            reason="i2_full_eval_failed",
        )
    registry.update_status(bad_candidate.id, StrategyStatus.REJECTED)

    _assert_dreamcycle_triggered_and_ran_arena(dreamcycle)
    _assert_full_eval_persisted(good_eval, task_set, experiment_store, task_store)
    _assert_full_eval_persisted(bad_eval, task_set, experiment_store, task_store)
    assert good_eval.verdict is FullEvalVerdict.PROMOTE_READY
    assert good_gate.approved is True
    assert promotion.strategy.status is StrategyStatus.SHADOW
    assert registry.get(good_candidate.id).status is StrategyStatus.SHADOW
    assert bad_eval.verdict is FullEvalVerdict.REJECT
    assert bad_gate.approved is False
    assert "win_rate_below_threshold" in bad_gate.failed_checks
    assert registry.get(bad_candidate.id).status is StrategyStatus.REJECTED
    assert registry.get(champion.id).status is StrategyStatus.CHAMPION
    assert [event.to_status for event in controller.list_events()] == [
        StrategyStatus.SHADOW
    ]


@dataclass(frozen=True)
class StrategyQualityRunner:
    quality_by_strategy_id: dict[str, float]
    default_quality: float = 0.60
    cost: float = 0.03
    latency_ms: int = 900
    token_count: int = 320

    def run(self, strategy: Strategy, task: Task) -> AgentRunResult:
        quality = self.quality_by_strategy_id.get(strategy.id, self.default_quality)
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
                    "name": "strategy_quality_runner",
                    "strategy_id": strategy.id,
                    "task_id": task.id,
                }
            ],
        )


def _stores(
    tmp_path: Path,
) -> tuple[StrategyRegistry, TaskStore, EvalExperimentStore, TaskSet]:
    database_path = tmp_path / "goagentx.db"
    task_set = load_task_set(TASK_SET_PATH)
    task_store = TaskStore(database_path)
    task_store.save_task_set(task_set)
    return (
        StrategyRegistry(database_path),
        task_store,
        EvalExperimentStore(database_path),
        task_set,
    )


def _save_degraded_history(
    task_store: TaskStore,
    *,
    champion_id: str,
    tasks: list[Task],
) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    scores = [0.82, 0.82, 0.82, 0.82, 0.50, 0.50, 0.50]
    for index, score in enumerate(scores):
        task = tasks[index % len(tasks)]
        task_store.save_run(
            TaskRun(
                id=f"history-{champion_id}-{index:03d}",
                task_id=task.id,
                strategy_id=champion_id,
                output_json={"history_score": score},
                score=score,
                score_breakdown={"total": score},
                success=True,
                cost=0.03,
                latency_ms=900,
                token_count=320,
                created_at=base_time + timedelta(minutes=index),
            )
        )


def _evolution_settings() -> EvolutionSettings:
    return EvolutionSettings(
        degradation_window=3,
        baseline_window=4,
        degradation_threshold=0.15,
    )


def _run_full_eval(
    *,
    champion: Strategy,
    candidate: Strategy,
    task_set: TaskSet,
    scorer: Scorer,
    settings: Settings,
    quality_by_strategy_id: dict[str, float],
    experiment_id: str,
    tmp_path: Path,
    experiment_store: EvalExperimentStore,
    task_store: TaskStore,
) -> FullEvaluationResult:
    return run_full_eval_from_settings(
        champion=champion,
        candidate=candidate,
        tasks=task_set.tasks,
        champion_runner=StrategyQualityRunner(quality_by_strategy_id),
        scorer=scorer,
        arena_settings=settings.arena,
        gate_settings=settings.promotion_gate,
        task_set_id=task_set.id,
        experiment_id=experiment_id,
        report_directory=tmp_path / "reports",
        experiment_store=experiment_store,
        task_store=task_store,
    )


def _assert_dreamcycle_triggered_and_ran_arena(result: DreamCycleResult) -> None:
    audit_events = _audit_event_types(result.audit_log_path)

    assert result.triggered is True
    assert result.reason == "score_drop_threshold_exceeded"
    assert result.detection.recent_sample_count == 3
    assert result.detection.baseline_sample_count == 4
    assert len(result.candidates) == 2
    assert all(item.quick_reject is not None for item in result.candidates)
    assert all(
        item.quick_reject.quick_reject_passed is True
        for item in result.candidates
        if item.quick_reject is not None
    )
    assert audit_events.count("candidate_created") == 2
    assert audit_events.count("quick_reject_completed") == 2
    assert audit_events[-1] == "dreamcycle_completed"


def _assert_full_eval_persisted(
    result: FullEvaluationResult,
    task_set: TaskSet,
    experiment_store: EvalExperimentStore,
    task_store: TaskStore,
) -> None:
    stored_experiment = experiment_store.get(result.experiment_id)
    stored_runs = task_store.list_recent_runs(
        limit=20,
        experiment_id=result.experiment_id,
    )

    assert result.report_path is not None
    assert result.report_path.exists()
    assert stored_experiment.report_path == result.report_path
    assert stored_experiment.verdict == result.verdict.value
    assert len(stored_runs) == len(task_set.tasks) * 2
    assert set(result.selected_task_ids) == {task.id for task in task_set.tasks}


def _audit_event_types(path: Path) -> list[str]:
    return [
        json.loads(line)["event_type"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
