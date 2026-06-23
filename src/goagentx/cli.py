"""Command-line interface for GoAgentX."""

from __future__ import annotations

import typer

from goagentx import __version__

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


def main() -> None:
    """Run the GoAgentX CLI application."""
    app()
