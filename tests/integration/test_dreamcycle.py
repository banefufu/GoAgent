import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from goagentx.config.settings import ArenaSettings, EvolutionSettings, load_settings
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
from goagentx.core.task import Task, TaskRun
from goagentx.evolution.dreamcycle import run_dreamcycle
from goagentx.evolution.mutation import StrategyMutator, load_mutation_settings
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore


def test_dreamcycle_generates_candidates_and_runs_quick_reject(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    champion = registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    _save_history(
        task_store,
        strategy_id=champion.id,
        baseline_scores=[0.8, 0.8, 0.8, 0.8],
        recent_scores=[0.68, 0.68, 0.68],
    )
    settings = load_settings()

    result = run_dreamcycle(
        champion_id=champion.id,
        registry=registry,
        task_store=task_store,
        mutator=StrategyMutator(load_mutation_settings(), seed=11),
        scorer=Scorer(settings.scoring),
        runner=QualityRunner({task.id: 0.9 for task in task_store.list_tasks()}),
        candidate_runner=QualityRunner(
            {task.id: 0.2 for task in task_store.list_tasks()}
        ),
        evolution_settings=_evolution_settings(),
        arena_settings=_arena_settings(),
        audit_log_path=tmp_path / "audit" / "dreamcycle.jsonl",
        candidate_count=3,
        auto_run_arena=True,
    )

    registry_candidates = registry.list_by_status(StrategyStatus.CANDIDATE)
    audit_events = _audit_event_types(result.audit_log_path)

    assert result.triggered is True
    assert result.reason == "score_drop_threshold_exceeded"
    assert len(result.candidates) == 3
    assert len(registry_candidates) == 3
    assert {candidate.parent_ids[0] for candidate in registry_candidates} == {
        champion.id
    }
    assert all(item.quick_reject is not None for item in result.candidates)
    assert all(
        item.quick_reject.quick_reject_passed is False
        for item in result.candidates
        if item.quick_reject is not None
    )
    assert audit_events.count("candidate_created") == 3
    assert audit_events.count("quick_reject_completed") == 3
    assert audit_events[-1] == "dreamcycle_completed"


def test_dreamcycle_skips_when_degradation_does_not_trigger(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    champion = registry.create(_strategy("champion-docs", StrategyStatus.CHAMPION))
    _save_history(
        task_store,
        strategy_id=champion.id,
        baseline_scores=[0.8, 0.8, 0.8, 0.8],
        recent_scores=[0.78, 0.78, 0.78],
    )
    settings = load_settings()

    result = run_dreamcycle(
        champion_id=champion.id,
        registry=registry,
        task_store=task_store,
        mutator=StrategyMutator(load_mutation_settings(), seed=11),
        scorer=Scorer(settings.scoring),
        runner=QualityRunner({task.id: 0.9 for task in task_store.list_tasks()}),
        evolution_settings=_evolution_settings(),
        arena_settings=_arena_settings(),
        audit_log_path=tmp_path / "audit" / "dreamcycle.jsonl",
        candidate_count=3,
        auto_run_arena=True,
    )

    assert result.triggered is False
    assert result.candidates == []
    assert registry.list_by_status(StrategyStatus.CANDIDATE) == []
    assert "dreamcycle_skipped" in _audit_event_types(result.audit_log_path)


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


def _save_history(
    task_store: TaskStore,
    *,
    strategy_id: str,
    baseline_scores: list[float],
    recent_scores: list[float],
) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    for index, score in enumerate(baseline_scores + recent_scores):
        task_id = f"task-doc-{index:03d}"
        task_store.save_task(
            Task(
                id=task_id,
                task_type="doc_qa",
                bucket="critical" if index % 3 == 0 else "baseline",
                input_json={"question": "What does GoAgentX evolve?"},
                expected_json={"contains": ["agent strategies"]},
                tags=["doc_qa"],
                created_at=base_time,
            )
        )
        task_store.save_run(
            TaskRun(
                id=f"run-{task_id}",
                task_id=task_id,
                strategy_id=strategy_id,
                output_json={"answer": "GoAgentX evolves agent strategies."},
                score=score,
                score_breakdown={"total": score},
                success=True,
                cost=0.03,
                latency_ms=900,
                token_count=320,
                created_at=base_time + timedelta(minutes=index),
            )
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


def _evolution_settings() -> EvolutionSettings:
    return EvolutionSettings(
        degradation_window=3,
        baseline_window=4,
        degradation_threshold=0.15,
    )


def _arena_settings() -> ArenaSettings:
    return ArenaSettings(
        quick_reject_rounds=5,
        full_eval_rounds=50,
        min_win_rate=0.55,
        p_value_threshold=0.05,
    )


def _audit_event_types(path: Path) -> list[str]:
    return [
        json.loads(line)["event_type"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
