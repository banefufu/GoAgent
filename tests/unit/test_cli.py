from typer.testing import CliRunner

from goagentx import __version__
from goagentx.cli import app


runner = CliRunner()


def test_cli_help_displays_application_name() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "GoAgentX" in result.output
    assert "--version" in result.output


def test_cli_version_displays_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"GoAgentX {__version__}" in result.output
