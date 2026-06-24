"""Command-line interface for GoAgentX."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from goagentx import __version__
from goagentx.adapters.agent_runner import FakeAgentRunner
from goagentx.arena.report import FullEvalError, run_full_eval_from_settings
from goagentx.config.settings import DEFAULT_CONFIG_DIR, load_settings
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.core.task import Task, TaskModelError, load_task_set
from goagentx.evolution.crossover import StrategyCrossover
from goagentx.evolution.dreamcycle import DreamCycleError, run_dreamcycle
from goagentx.evolution.genome_ga import GenomeGAError, GenomeGASettings, run_genome_ga
from goagentx.evolution.mutation import StrategyMutator, load_mutation_settings
from goagentx.promotion.controller import PromotionController, PromotionControllerError
from goagentx.promotion.gate import PromotionGateMetrics, evaluate_promotion_gate
from goagentx.promotion.rollback import RollbackController, RollbackControllerError
from goagentx.registry.db import initialize_database
from goagentx.registry.experiment_store import EvalExperimentStore
from goagentx.registry.strategy_io import (
    StrategyIOError,
    export_strategy_yaml,
    import_strategy_yaml,
    strategy_to_yaml_data,
)
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.strategy_registry import StrategyRegistryError
from goagentx.registry.task_store import TaskStore

app = typer.Typer(
    help="GoAgentX strategy evolution and evaluation CLI.",
    no_args_is_help=True,
)
evolve_app = typer.Typer(
    help="Generate candidate strategies with evolutionary workflows.",
    no_args_is_help=True,
)
strategy_app = typer.Typer(
    help="Inspect and move Strategy YAML files through the registry.",
    no_args_is_help=True,
)
app.add_typer(evolve_app, name="evolve")
app.add_typer(strategy_app, name="strategy")


@app.callback(invoke_without_command=True)
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the installed GoAgentX version and exit.",
    ),
) -> None:
    """Configure global CLI options."""
    if version:
        typer.echo(f"GoAgentX {__version__}")
        raise typer.Exit()


@app.command("init")
def init_command(
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
) -> None:
    """Initialize the local GoAgentX SQLite database."""
    settings = load_settings(config_dir)
    target_path = database_path or settings.database.path
    initialized_path = initialize_database(target_path)
    typer.echo(f"Initialized GoAgentX database at {initialized_path}")


@app.command("eval")
def eval_command(
    champion_id: Annotated[
        str,
        typer.Option(
            "--champion",
            help="Champion strategy id.",
        ),
    ],
    candidate_id: Annotated[
        str,
        typer.Option(
            "--candidate",
            help="Candidate strategy id.",
        ),
    ],
    task_set: Annotated[
        str,
        typer.Option(
            "--task-set",
            help="Task set id already in the database, or a JSON task-set file path.",
        ),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    report_dir: Annotated[
        Path | None,
        typer.Option(
            "--report-dir",
            help="Override the configured report output directory.",
        ),
    ] = None,
    experiment_id: Annotated[
        str | None,
        typer.Option(
            "--experiment-id",
            help="Explicit Full Eval experiment id.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for deterministic task selection and stats.",
        ),
    ] = 0,
) -> None:
    """Run Full Eval for a champion/candidate pair and write a report."""
    settings = load_settings(config_dir)
    database = database_path or settings.database.path
    registry = StrategyRegistry(database)
    task_store = TaskStore(database)
    experiment_store = EvalExperimentStore(database)

    try:
        champion = registry.get(champion_id)
        candidate = registry.get(candidate_id)
        tasks, task_set_id = _resolve_eval_tasks(task_store, task_set)
        result = run_full_eval_from_settings(
            champion=champion,
            candidate=candidate,
            tasks=tasks,
            champion_runner=FakeAgentRunner(),
            candidate_runner=FakeAgentRunner(),
            scorer=Scorer(settings.scoring),
            arena_settings=settings.arena,
            gate_settings=settings.promotion_gate,
            task_set_id=task_set_id,
            experiment_id=experiment_id,
            report_directory=report_dir or settings.reports.directory,
            experiment_store=experiment_store,
            task_store=task_store,
            seed=seed,
        )
    except (
        FullEvalError,
        StrategyRegistryError,
        TaskModelError,
        ValueError,
    ) as exc:
        typer.echo(f"Eval failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Full Eval verdict: {result.verdict.value}")
    typer.echo(f"Experiment: {result.experiment_id}")
    typer.echo(f"Task set: {result.task_set_id}")
    typer.echo(f"Selected tasks: {len(result.selected_task_ids)}")
    typer.echo(f"Win rate: {result.evaluation.win_rate:.4f}")
    typer.echo(f"Avg score delta: {result.evaluation.avg_score_delta:.4f}")
    typer.echo(f"Report: {result.report_path}")
    if result.failed_checks:
        typer.echo("Failed checks:")
        for check in result.failed_checks:
            typer.echo(f"- {check}")


@app.command("promote")
def promote_command(
    candidate_id: Annotated[
        str,
        typer.Option(
            "--candidate",
            help="Candidate strategy id to promote.",
        ),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Target promotion status: shadow, canary, or champion.",
        ),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    experiment_id: Annotated[
        str | None,
        typer.Option(
            "--experiment-id",
            help="Full Eval experiment id or manual approval label.",
        ),
    ] = None,
    champion_id: Annotated[
        str | None,
        typer.Option(
            "--champion",
            help="Champion id used for gate metrics. Defaults to current domain champion.",
        ),
    ] = None,
    win_rate: Annotated[
        float,
        typer.Option(
            "--win-rate",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = 1.0,
    p_value: Annotated[
        float,
        typer.Option(
            "--p-value",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = 0.0,
    avg_score_delta: Annotated[
        float,
        typer.Option(
            "--avg-score-delta",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = 0.1,
    cost_delta: Annotated[
        float,
        typer.Option(
            "--cost-delta",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = 0.0,
    latency_delta: Annotated[
        float,
        typer.Option(
            "--latency-delta",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = 0.0,
    safety_violation_count: Annotated[
        int,
        typer.Option(
            "--safety-violation-count",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = 0,
    critical_bucket_regression: Annotated[
        bool,
        typer.Option(
            "--critical-bucket-regression/--no-critical-bucket-regression",
            help="Gate metric used for manual promotion approval.",
        ),
    ] = False,
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason",
            help="Audit reason for the promotion event.",
        ),
    ] = None,
) -> None:
    """Promote a candidate through shadow, canary, and champion states."""
    settings = load_settings(config_dir)
    registry = StrategyRegistry(database_path or settings.database.path)
    try:
        candidate = registry.get(candidate_id)
        resolved_target = StrategyStatus(mode)
        resolved_champion_id = champion_id or registry.get_champion(candidate.task_type).id
        gate = evaluate_promotion_gate(
            PromotionGateMetrics(
                experiment_id=experiment_id or f"manual-promotion-{candidate.id}",
                champion_id=resolved_champion_id,
                candidate_id=candidate.id,
                win_rate=win_rate,
                p_value=p_value,
                avg_score_delta=avg_score_delta,
                cost_delta=cost_delta,
                latency_delta=latency_delta,
                safety_violation_count=safety_violation_count,
                critical_bucket_regression=critical_bucket_regression,
            ),
            settings.promotion_gate,
        )
        if not gate.approved:
            typer.echo(
                "Promotion gate rejected: " + ", ".join(gate.failed_checks),
                err=True,
            )
            raise typer.Exit(code=1)
        result = PromotionController(registry).promote(
            candidate.id,
            target_status=resolved_target,
            gate=gate,
            reason=reason,
        )
    except (
        PromotionControllerError,
        StrategyRegistryError,
        ValueError,
    ) as exc:
        typer.echo(f"Promotion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Promoted {result.strategy.id}: "
        f"{result.event.from_status.value} -> {result.event.to_status.value}"
    )
    typer.echo(f"Gate decision: {result.gate.decision.value}")
    typer.echo(f"Event: {result.event.id}")


@app.command("rollback")
def rollback_command(
    to_strategy_id: Annotated[
        str,
        typer.Option(
            "--to",
            help="Stable strategy id to restore.",
        ),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    failed_strategy_id: Annotated[
        str | None,
        typer.Option(
            "--failed",
            help="Failed shadow, canary, or champion strategy id.",
        ),
    ] = None,
    failed_status: Annotated[
        str,
        typer.Option(
            "--failed-status",
            help="Status assigned to the failed strategy: rolled_back or retired.",
        ),
    ] = StrategyStatus.ROLLED_BACK.value,
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason",
            help="Audit reason for rollback events.",
        ),
    ] = None,
) -> None:
    """Rollback to a stable strategy and audit the status changes."""
    settings = load_settings(config_dir)
    registry = StrategyRegistry(database_path or settings.database.path)
    try:
        result = RollbackController(registry).rollback(
            to_strategy_id,
            failed_strategy_id=failed_strategy_id,
            failed_status=StrategyStatus(failed_status),
            reason=reason,
        )
    except (RollbackControllerError, StrategyRegistryError, ValueError) as exc:
        typer.echo(f"Rollback failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Rollback restored {result.restored_strategy.id} "
        f"as {result.restored_strategy.status.value}."
    )
    if result.failed_strategy is not None:
        typer.echo(
            f"Failed strategy {result.failed_strategy.id} "
            f"marked {result.failed_strategy.status.value}."
        )
    typer.echo(f"Events written: {len(result.events)}")


@strategy_app.command("list")
def strategy_list_command(
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Restrict output to one lifecycle status.",
        ),
    ] = None,
    task_type: Annotated[
        str | None,
        typer.Option(
            "--task-type",
            help="Restrict output to one task type.",
        ),
    ] = None,
) -> None:
    """List registered strategies."""
    registry = _strategy_registry(config_dir, database_path)
    try:
        strategies = _list_strategies(registry, status=status)
    except ValueError as exc:
        typer.echo(f"Strategy list failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if task_type is not None:
        strategies = [strategy for strategy in strategies if strategy.task_type == task_type]

    if not strategies:
        typer.echo("No strategies found.")
        return

    typer.echo("id\tstatus\tversion\ttask_type\tname")
    for strategy in strategies:
        typer.echo(
            "\t".join(
                [
                    strategy.id,
                    strategy.status.value,
                    str(strategy.version),
                    strategy.task_type or "-",
                    strategy.name,
                ]
            )
        )


@strategy_app.command("show")
def strategy_show_command(
    strategy_id: Annotated[
        str,
        typer.Argument(help="Strategy id to display."),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
) -> None:
    """Show one registered strategy as YAML."""
    registry = _strategy_registry(config_dir, database_path)
    try:
        strategy = registry.get(strategy_id)
    except StrategyRegistryError as exc:
        typer.echo(f"Strategy show failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        yaml.safe_dump(
            strategy_to_yaml_data(strategy),
            sort_keys=False,
            allow_unicode=True,
        ),
        nl=False,
    )


@strategy_app.command("import")
def strategy_import_command(
    source_path: Annotated[
        Path,
        typer.Argument(help="Strategy YAML file to import."),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    status: Annotated[
        str,
        typer.Option(
            "--status",
            help="Imported lifecycle status: draft or candidate.",
        ),
    ] = StrategyStatus.CANDIDATE.value,
) -> None:
    """Import a Strategy YAML file as draft or candidate."""
    registry = _strategy_registry(config_dir, database_path)
    try:
        imported = import_strategy_yaml(registry, source_path, status=status)
    except (StrategyIOError, StrategyRegistryError, ValueError) as exc:
        typer.echo(f"Strategy import failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Imported strategy {imported.id} as {imported.status.value}.")


@strategy_app.command("export")
def strategy_export_command(
    strategy_id: Annotated[
        str,
        typer.Argument(help="Strategy id to export."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Destination YAML path.",
        ),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
) -> None:
    """Export one registered strategy to YAML."""
    registry = _strategy_registry(config_dir, database_path)
    try:
        strategy = registry.get(strategy_id)
        written_path = export_strategy_yaml(strategy, output_path)
    except (StrategyIOError, StrategyRegistryError) as exc:
        typer.echo(f"Strategy export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Exported strategy {strategy.id} to {written_path}.")


@evolve_app.command("ga")
def evolve_ga_command(
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    task_type: Annotated[
        str | None,
        typer.Option(
            "--task-type",
            help="Restrict parent selection to a task type.",
        ),
    ] = None,
    population: Annotated[
        str | None,
        typer.Option(
            "--population",
            help="Optional population label used as the generated candidate id prefix.",
        ),
    ] = None,
    population_size: Annotated[
        int,
        typer.Option(
            "--population-size",
            help="Target next-generation population size, including elites.",
        ),
    ] = 20,
    elite_ratio: Annotated[
        float,
        typer.Option(
            "--elite-ratio",
            help="Fraction of the population reserved for existing elite strategies.",
        ),
    ] = 0.2,
    mutation_rate: Annotated[
        float,
        typer.Option(
            "--mutation-rate",
            help="Fraction of generated candidates created by mutation.",
        ),
    ] = 0.2,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for deterministic parent and operation selection.",
        ),
    ] = 0,
    candidate_id_prefix: Annotated[
        str | None,
        typer.Option(
            "--candidate-id-prefix",
            help="Explicit generated candidate id prefix.",
        ),
    ] = None,
) -> None:
    """Generate a Genome GA candidate population."""
    settings = load_settings(config_dir)
    database = database_path or settings.database.path
    mutation_settings = load_mutation_settings(config_dir)
    prefix = candidate_id_prefix or (f"ga-{population}" if population else None)

    try:
        result = run_genome_ga(
            registry=StrategyRegistry(database),
            task_store=TaskStore(database),
            mutator=StrategyMutator(mutation_settings, seed=seed),
            crossover=StrategyCrossover(mutation_settings, seed=seed),
            settings=GenomeGASettings(
                population_size=population_size,
                elite_ratio=elite_ratio,
                mutation_rate=mutation_rate,
                candidate_id_prefix=prefix,
            ),
            task_type=task_type,
            seed=seed,
        )
    except GenomeGAError as exc:
        typer.echo(f"Genome GA failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        "Generated "
        f"{len(result.candidate_pool)} candidates "
        f"with {len(result.elite_pool)} elites retained."
    )
    for candidate_id in result.candidate_ids:
        typer.echo(f"- {candidate_id}")


@evolve_app.command("dream")
def evolve_dream_command(
    strategy_id: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Champion strategy id to evolve from.",
        ),
    ],
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            help="Directory containing GoAgentX YAML configuration files.",
        ),
    ] = DEFAULT_CONFIG_DIR,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--database-path",
            help="Override the configured SQLite database path.",
        ),
    ] = None,
    task_type: Annotated[
        str | None,
        typer.Option(
            "--task-type",
            help="Restrict degradation detection and Arena tasks to a task type.",
        ),
    ] = None,
    task_set: Annotated[
        str | None,
        typer.Option(
            "--task-set",
            help="Optional task set id already in the database, or a JSON task-set file path.",
        ),
    ] = None,
    candidate_count: Annotated[
        int,
        typer.Option(
            "--candidate-count",
            help="Number of candidates to generate, from 1 to 3.",
        ),
    ] = 3,
    auto_run_arena: Annotated[
        bool,
        typer.Option(
            "--auto-run-arena/--no-auto-run-arena",
            help="Run Quick Reject for each generated candidate.",
        ),
    ] = True,
    manual_trigger: Annotated[
        bool,
        typer.Option(
            "--manual-trigger/--require-degradation",
            help="Generate candidates immediately, or require degradation detection first.",
        ),
    ] = True,
    audit_log_path: Annotated[
        Path | None,
        typer.Option(
            "--audit-log",
            help="DreamCycle JSONL audit log path.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for mutation and Quick Reject selection.",
        ),
    ] = 0,
) -> None:
    """Generate DreamCycle candidates and optionally run Quick Reject."""
    settings = load_settings(config_dir)
    database = database_path or settings.database.path
    registry = StrategyRegistry(database)
    task_store = TaskStore(database)
    mutation_settings = load_mutation_settings(config_dir)

    if task_set is not None:
        try:
            _resolve_eval_tasks(task_store, task_set)
        except (TaskModelError, ValueError) as exc:
            typer.echo(f"DreamCycle failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    resolved_audit_log_path = audit_log_path or (
        settings.reports.directory / f"dreamcycle-{strategy_id}.jsonl"
    )
    try:
        result = run_dreamcycle(
            champion_id=strategy_id,
            registry=registry,
            task_store=task_store,
            mutator=StrategyMutator(mutation_settings, seed=seed),
            scorer=Scorer(settings.scoring),
            runner=FakeAgentRunner(),
            candidate_runner=FakeAgentRunner(),
            evolution_settings=settings.evolution,
            arena_settings=settings.arena,
            audit_log_path=resolved_audit_log_path,
            task_type=task_type,
            candidate_count=candidate_count,
            auto_run_arena=auto_run_arena,
            manual_trigger=manual_trigger,
            seed=seed,
        )
    except (DreamCycleError, StrategyRegistryError, ValueError) as exc:
        typer.echo(f"DreamCycle failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"DreamCycle triggered: {str(result.triggered).lower()}")
    typer.echo(f"Reason: {result.reason}")
    typer.echo(f"Audit log: {result.audit_log_path}")
    typer.echo(f"Generated candidates: {len(result.candidates)}")
    for candidate in result.candidates:
        typer.echo(
            f"- {candidate.candidate.id} "
            f"mutation={candidate.mutation_kind.value}"
        )
        if candidate.quick_reject is not None:
            typer.echo(
                "  quick_reject="
                f"{candidate.quick_reject.decision.value}"
            )


def _strategy_registry(
    config_dir: Path,
    database_path: Path | None,
) -> StrategyRegistry:
    settings = load_settings(config_dir)
    return StrategyRegistry(database_path or settings.database.path)


def _list_strategies(
    registry: StrategyRegistry,
    *,
    status: str | None,
) -> list[Strategy]:
    if status is not None:
        return registry.list_by_status(StrategyStatus(status))

    strategies: list[Strategy] = []
    for strategy_status in StrategyStatus:
        strategies.extend(registry.list_by_status(strategy_status))
    return sorted(strategies, key=lambda strategy: (strategy.created_at, strategy.id))


def _resolve_eval_tasks(
    task_store: TaskStore,
    task_set: str,
) -> tuple[list[Task], str]:
    task_set_path = Path(task_set)
    if task_set_path.exists():
        loaded = load_task_set(task_set_path)
        return task_store.save_task_set(loaded), loaded.id

    tasks = task_store.list_tasks(task_set_id=task_set)
    if not tasks:
        raise ValueError(f"Task set not found: {task_set}")
    return tasks, task_set


def main() -> None:
    """Run the GoAgentX CLI application."""
    app()
