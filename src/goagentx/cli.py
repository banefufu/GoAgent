"""Command-line interface for GoAgentX."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from goagentx import __version__
from goagentx.config.settings import DEFAULT_CONFIG_DIR, load_settings
from goagentx.registry.db import initialize_database

app = typer.Typer(
    help="GoAgentX strategy evolution and evaluation CLI.",
    no_args_is_help=True,
)


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


def main() -> None:
    """Run the GoAgentX CLI application."""
    app()
