"""SQLite initialization utilities for GoAgentX."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

SCHEMA_RESOURCE = "schema.sql"


class DatabaseInitializationError(RuntimeError):
    """Raised when the local SQLite database cannot be initialized."""


def initialize_database(database_path: str | Path) -> Path:
    """Create or migrate the local GoAgentX SQLite database.

    Args:
        database_path: Target SQLite database path.

    Returns:
        The normalized path that was initialized.

    Raises:
        DatabaseInitializationError: If SQLite rejects the schema or path.
    """
    target_path = Path(database_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = load_schema_sql()

    try:
        with sqlite3.connect(target_path) as connection:
            connection.executescript(schema_sql)
            _ensure_strategy_schema(connection)
            connection.commit()
    except sqlite3.Error as exc:
        raise DatabaseInitializationError(
            f"Failed to initialize GoAgentX database at {target_path}: {exc}"
        ) from exc

    return target_path


def load_schema_sql() -> str:
    """Load the packaged SQLite schema."""
    return resources.files("goagentx.registry").joinpath(SCHEMA_RESOURCE).read_text(
        encoding="utf-8"
    )


def _ensure_strategy_schema(connection: sqlite3.Connection) -> None:
    """Apply small migrations and indexes for the strategies table."""
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(strategies)").fetchall()
    }
    if "task_type" not in columns:
        connection.execute("ALTER TABLE strategies ADD COLUMN task_type TEXT")
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategies_single_champion_task_type
        ON strategies(task_type)
        WHERE status = 'champion' AND task_type IS NOT NULL
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategies_single_global_champion
        ON strategies(status)
        WHERE status = 'champion' AND task_type IS NULL
        """
    )
