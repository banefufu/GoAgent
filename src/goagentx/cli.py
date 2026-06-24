"""Command-line interface for GoAgentX."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from goagentx import __version__
from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.config.settings import DEFAULT_CONFIG_DIR, load_settings
from goagentx.evolution.crossover import StrategyCrossover
from goagentx.evolution.genome_ga import GenomeGAError, GenomeGASettings, run_genome_ga
from goagentx.evolution.mutation import StrategyMutator, load_mutation_settings
from goagentx.registry.db import initialize_database
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


def main() -> None:
    """Run the GoAgentX CLI application."""
    app()
