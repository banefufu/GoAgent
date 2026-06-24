from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from goagentx.cli import app
from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolsGenome,
)
from goagentx.core.task import Task, TaskRun
from goagentx.evolution.crossover import StrategyCrossover
from goagentx.evolution.genome_ga import GenomeGAError, GenomeGASettings, run_genome_ga
from goagentx.evolution.mutation import StrategyMutator, load_mutation_settings
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore


runner = CliRunner()


def test_genome_ga_generates_candidate_population_without_running_arena(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    parents = [
        registry.create(
            _strategy(
                "parent-high",
                StrategyStatus.RETIRED,
                temperature=0.2,
                role="senior_code_reviewer",
            )
        ),
        registry.create(
            _strategy(
                "parent-mid",
                StrategyStatus.SHADOW,
                temperature=0.4,
                role="cautious_debugger",
            )
        ),
        registry.create(
            _strategy(
                "parent-ok",
                StrategyStatus.CANARY,
                temperature=0.7,
                role="production_incident_analyst",
            )
        ),
        registry.create(_strategy("parent-low", StrategyStatus.RETIRED, temperature=0.9)),
        registry.create(_strategy("parent-worst", StrategyStatus.RETIRED, temperature=1.1)),
    ]
    scores = {
        "parent-high": [0.95, 0.94],
        "parent-mid": [0.88, 0.87],
        "parent-ok": [0.8, 0.79],
        "parent-low": [0.3, 0.31],
        "parent-worst": [0.2, 0.19],
    }
    for parent in parents:
        _save_history(task_store, strategy_id=parent.id, scores=scores[parent.id])
    mutation_settings = load_mutation_settings()

    result = run_genome_ga(
        registry=registry,
        task_store=task_store,
        mutator=StrategyMutator(mutation_settings, seed=5),
        crossover=StrategyCrossover(mutation_settings, seed=5),
        settings=GenomeGASettings(
            population_size=6,
            elite_ratio=0.33,
            mutation_rate=0.5,
            candidate_id_prefix="test-ga",
        ),
        task_type="doc_qa",
        seed=7,
    )

    registry_candidates = registry.list_by_status(StrategyStatus.CANDIDATE)
    parent_ids = result.parent_selection.parent_ids

    assert result.reason == "generated"
    assert result.elite_ids == ["parent-high", "parent-mid"]
    assert len(result.candidate_pool) == 4
    assert len(result.population) == 6
    assert result.mutation_count == 2
    assert result.crossover_count == 2
    assert result.candidate_ids == [candidate.id for candidate in registry_candidates]
    assert "parent-low" not in parent_ids
    assert "parent-worst" not in parent_ids
    assert all(candidate.status is StrategyStatus.CANDIDATE for candidate in registry_candidates)
    assert all(candidate.task_type == "doc_qa" for candidate in registry_candidates)
    assert all(set(candidate.parent_ids) <= set(parent_ids) for candidate in registry_candidates)
    assert all(
        task_store.list_recent_runs(strategy_id=candidate.id, limit=1) == []
        for candidate in registry_candidates
    )


def test_genome_ga_rejects_insufficient_scored_parent_pool(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    scored_parent = registry.create(_strategy("parent-only", StrategyStatus.RETIRED))
    registry.create(_strategy("parent-no-history", StrategyStatus.SHADOW))
    _save_history(task_store, strategy_id=scored_parent.id, scores=[0.9, 0.91])
    mutation_settings = load_mutation_settings()

    with pytest.raises(GenomeGAError, match="insufficient parent pool"):
        run_genome_ga(
            registry=registry,
            task_store=task_store,
            mutator=StrategyMutator(mutation_settings, seed=5),
            crossover=StrategyCrossover(mutation_settings, seed=5),
            settings=GenomeGASettings(population_size=4, candidate_id_prefix="bad-ga"),
            task_type="doc_qa",
            seed=7,
        )


def test_cli_evolve_ga_generates_candidates(tmp_path: Path) -> None:
    database_path = tmp_path / "goagentx.db"
    registry = StrategyRegistry(database_path)
    task_store = TaskStore(database_path)
    for strategy_id, status, score in [
        ("parent-high", StrategyStatus.RETIRED, 0.95),
        ("parent-mid", StrategyStatus.SHADOW, 0.85),
        ("parent-ok", StrategyStatus.CANARY, 0.75),
    ]:
        registry.create(_strategy(strategy_id, status))
        _save_history(task_store, strategy_id=strategy_id, scores=[score, score])

    result = runner.invoke(
        app,
        [
            "evolve",
            "ga",
            "--config-dir",
            "configs",
            "--database-path",
            str(database_path),
            "--task-type",
            "doc_qa",
            "--population-size",
            "4",
            "--population",
            "sample",
            "--mutation-rate",
            "0.34",
            "--seed",
            "3",
        ],
    )

    registry_candidates = registry.list_by_status(StrategyStatus.CANDIDATE)

    assert result.exit_code == 0, result.output
    assert "Generated 3 candidates" in result.output
    assert len(registry_candidates) == 3
    assert all(candidate.id.startswith("ga-sample-") for candidate in registry_candidates)
    assert all(candidate.status is StrategyStatus.CANDIDATE for candidate in registry_candidates)


def _strategy(
    strategy_id: str,
    status: StrategyStatus,
    *,
    temperature: float = 0.4,
    role: str = "senior_code_reviewer",
) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type="doc_qa",
        status=status,
        genome=Genome(
            model=ModelGenome(
                provider="openai_compatible",
                name="gpt-4.1",
                temperature=temperature,
                top_p=0.9,
            ),
            prompt_genome=PromptGenome(
                role=role,
                reasoning_style="evidence_first",
                risk_policy="strict",
                output_format="findings_first",
            ),
            tools=ToolsGenome(enabled=["repo_search", "shell_readonly", "browser"]),
        ),
    )


def _save_history(
    task_store: TaskStore,
    *,
    strategy_id: str,
    scores: list[float],
) -> None:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    for index, score in enumerate(scores):
        task_id = f"task-{strategy_id}-{index:03d}"
        created_at = base_time + timedelta(minutes=index)
        task_store.save_task(
            Task(
                id=task_id,
                task_type="doc_qa",
                bucket="baseline",
                input_json={"question": "What does GoAgentX evolve?"},
                expected_json={"contains": ["agent strategies"]},
                tags=["doc_qa"],
                created_at=created_at,
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
                created_at=created_at,
            )
        )
