"""Command-line interface for GoAgentX."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from goagentx import __version__
from goagentx.config.settings import DEFAULT_CONFIG_DIR, load_settings
from goagentx.evolution.crossover import StrategyCrossover
from goagentx.evolution.genome_ga import GenomeGAError, GenomeGASettings, run_genome_ga
from goagentx.evolution.mutation import StrategyMutator, load_mutation_settings
from goagentx.registry.db import initialize_database
from goagentx.registry.strategy_registry import StrategyRegistry
from goagentx.registry.task_store import TaskStore

app = typer.Typer(
    help="GoAgentX strategy evolution and evaluation CLI.",
    no_args_is_help=True,
)
evolve_app = typer.Typer(
    help="Generate candidate strategies with evolutionary workflows.",
    no_args_is_help=True,
)
app.add_typer(evolve_app, name="evolve")


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


def main() -> None:
    """Run the GoAgentX CLI application."""
    app()
