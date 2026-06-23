import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from goagentx.cli import app
from goagentx.registry.db import initialize_database


EXPECTED_TABLES = {
    "strategies",
    "tasks",
    "task_runs",
    "eval_experiments",
    "promotion_events",
}


def test_initialize_database_creates_expected_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "goagentx.db"

    initialized_path = initialize_database(database_path)

    assert initialized_path == database_path
    assert database_path.exists()
    assert _table_names(database_path) >= EXPECTED_TABLES


def test_initialize_database_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "goagentx.db"

    first_path = initialize_database(database_path)
    second_path = initialize_database(database_path)

    assert first_path == second_path
    assert _table_names(database_path) >= EXPECTED_TABLES


def test_init_command_uses_configured_database_path(tmp_path: Path) -> None:
    config_dir = _write_config_set(tmp_path)
    database_path = tmp_path / "custom" / "goagentx.db"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            "--config-dir",
            str(config_dir),
            "--database-path",
            str(database_path),
        ],
    )

    assert result.exit_code == 0
    assert "Initialized GoAgentX database" in result.output
    assert database_path.exists()
    assert _table_names(database_path) >= EXPECTED_TABLES


def _table_names(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def _write_config_set(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "goagentx.yaml").write_text(
        """
database:
  path: data/goagentx.db
reports:
  directory: reports
evolution:
  degradation_window: 50
  baseline_window: 300
  degradation_threshold: 0.15
arena:
  quick_reject_rounds: 5
  full_eval_rounds: 50
  min_win_rate: 0.55
  p_value_threshold: 0.05
""".lstrip(),
        encoding="utf-8",
    )
    (config_dir / "scoring.yaml").write_text(
        """
weights:
  quality: 0.70
  cost: 0.10
  latency: 0.10
  safety: 0.10
safety_penalty: 1.0
""".lstrip(),
        encoding="utf-8",
    )
    (config_dir / "promotion_gate.yaml").write_text(
        """
min_win_rate: 0.55
max_p_value: 0.05
min_score_delta: 0.0
max_cost_delta: 0.20
max_latency_delta: 0.20
require_no_safety_violation: true
require_no_critical_bucket_regression: true
""".lstrip(),
        encoding="utf-8",
    )
    return config_dir
