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


def test_initialize_database_migrates_strategy_task_type_column(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "goagentx.db"
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE strategies (
              id TEXT PRIMARY KEY,
              version INTEGER NOT NULL,
              name TEXT NOT NULL,
              status TEXT NOT NULL,
              genome_json TEXT NOT NULL,
              parent_ids_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              notes TEXT
            )
            """
        )

    initialize_database(database_path)

    assert "task_type" in _column_names(database_path, "strategies")


def test_initialize_database_migrates_task_store_columns(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "goagentx.db"
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
              id TEXT PRIMARY KEY,
              task_type TEXT NOT NULL,
              bucket TEXT NOT NULL,
              input_json TEXT NOT NULL,
              expected_json TEXT,
              tags_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE task_runs (
              id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              strategy_id TEXT NOT NULL,
              experiment_id TEXT,
              output_json TEXT NOT NULL,
              score REAL NOT NULL,
              success INTEGER NOT NULL,
              cost REAL NOT NULL,
              latency_ms INTEGER NOT NULL,
              token_count INTEGER NOT NULL,
              tool_calls_json TEXT NOT NULL,
              error_json TEXT,
              created_at TEXT NOT NULL
            )
            """
        )

    initialize_database(database_path)

    assert "task_set_id" in _column_names(database_path, "tasks")
    assert "score_breakdown_json" in _column_names(database_path, "task_runs")


def test_initialize_database_migrates_eval_experiment_gate_columns(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "goagentx.db"
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE eval_experiments (
              id TEXT PRIMARY KEY,
              champion_id TEXT NOT NULL,
              candidate_id TEXT NOT NULL,
              task_set_id TEXT NOT NULL,
              quick_reject_passed INTEGER NOT NULL,
              win_rate REAL NOT NULL,
              p_value REAL,
              avg_score_delta REAL NOT NULL,
              cost_delta REAL NOT NULL,
              latency_delta REAL NOT NULL,
              verdict TEXT NOT NULL,
              report_path TEXT,
              created_at TEXT NOT NULL
            )
            """
        )

    initialize_database(database_path)

    assert "safety_violation_count" in _column_names(
        database_path,
        "eval_experiments",
    )
    assert "critical_bucket_regression" in _column_names(
        database_path,
        "eval_experiments",
    )


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


def _column_names(database_path: Path, table_name: str) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


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
normalization:
  max_cost: 1.0
  max_latency_ms: 10000
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
