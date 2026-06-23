"""SQLite-backed strategy registry."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.registry.db import initialize_database


class StrategyRegistryError(RuntimeError):
    """Base error for strategy registry failures."""


class StrategyAlreadyExistsError(StrategyRegistryError):
    """Raised when creating a strategy with a duplicate id."""


class StrategyNotFoundError(StrategyRegistryError):
    """Raised when a requested strategy does not exist."""


class StrategyRegistry:
    """Persist and query Strategy records in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        """Initialize the registry and ensure the SQLite schema exists."""
        self.database_path = initialize_database(database_path)

    def create(self, strategy: Strategy) -> Strategy:
        """Create a strategy record.

        If the new strategy is a champion, any existing champion for the same
        task type is retired in the same transaction.
        """
        with self._connect() as connection:
            try:
                if strategy.status is StrategyStatus.CHAMPION:
                    self._retire_existing_champion(
                        connection,
                        task_type=strategy.task_type,
                        excluding_strategy_id=strategy.id,
                    )
                connection.execute(
                    """
                    INSERT INTO strategies (
                      id, version, name, task_type, status, genome_json,
                      parent_ids_json, created_at, updated_at, notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _strategy_to_row(strategy),
                )
            except sqlite3.IntegrityError as exc:
                raise StrategyAlreadyExistsError(
                    f"Strategy already exists or violates registry constraints: {strategy.id}"
                ) from exc

        return self.get(strategy.id)

    def get(self, strategy_id: str) -> Strategy:
        """Return a strategy by id."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, version, name, task_type, status, genome_json,
                       parent_ids_json, created_at, updated_at, notes
                FROM strategies
                WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()

        if row is None:
            raise StrategyNotFoundError(f"Strategy not found: {strategy_id}")
        return _row_to_strategy(row)

    def list_by_status(self, status: str | StrategyStatus) -> list[Strategy]:
        """List strategies matching a lifecycle status."""
        strategy_status = StrategyStatus(status)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, version, name, task_type, status, genome_json,
                       parent_ids_json, created_at, updated_at, notes
                FROM strategies
                WHERE status = ?
                ORDER BY created_at ASC, id ASC
                """,
                (strategy_status.value,),
            ).fetchall()
        return [_row_to_strategy(row) for row in rows]

    def get_champion(self, task_type: str | None = None) -> Strategy:
        """Return the champion for a task type or the global champion."""
        where_clause = "task_type IS NULL" if task_type is None else "task_type = ?"
        parameters: tuple[str, ...] = () if task_type is None else (task_type,)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT id, version, name, task_type, status, genome_json,
                       parent_ids_json, created_at, updated_at, notes
                FROM strategies
                WHERE status = 'champion' AND {where_clause}
                ORDER BY updated_at DESC, id ASC
                LIMIT 1
                """,
                parameters,
            ).fetchone()

        if row is None:
            domain = "global" if task_type is None else task_type
            raise StrategyNotFoundError(f"Champion strategy not found for {domain}")
        return _row_to_strategy(row)

    def update_status(
        self,
        strategy_id: str,
        status: str | StrategyStatus,
    ) -> None:
        """Update a strategy status and refresh updated_at."""
        strategy_status = StrategyStatus(status)
        with self._connect() as connection:
            existing = self._get_row(connection, strategy_id)
            if existing is None:
                raise StrategyNotFoundError(f"Strategy not found: {strategy_id}")

            updated_at = _next_updated_at_iso(existing["updated_at"])
            task_type = existing["task_type"]
            if strategy_status is StrategyStatus.CHAMPION:
                self._retire_existing_champion(
                    connection,
                    task_type=task_type,
                    excluding_strategy_id=strategy_id,
                )

            connection.execute(
                """
                UPDATE strategies
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (strategy_status.value, updated_at, strategy_id),
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured for dict-like rows."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _get_row(
        self,
        connection: sqlite3.Connection,
        strategy_id: str,
    ) -> sqlite3.Row | None:
        """Return a raw strategy row by id."""
        return connection.execute(
            """
            SELECT id, version, name, task_type, status, genome_json,
                   parent_ids_json, created_at, updated_at, notes
            FROM strategies
            WHERE id = ?
            """,
            (strategy_id,),
        ).fetchone()

    def _retire_existing_champion(
        self,
        connection: sqlite3.Connection,
        task_type: str | None,
        excluding_strategy_id: str,
    ) -> None:
        """Retire an existing champion in the same task-type domain."""
        updated_at = _utc_now_iso()
        if task_type is None:
            connection.execute(
                """
                UPDATE strategies
                SET status = ?, updated_at = ?
                WHERE status = ? AND task_type IS NULL AND id != ?
                """,
                (
                    StrategyStatus.RETIRED.value,
                    updated_at,
                    StrategyStatus.CHAMPION.value,
                    excluding_strategy_id,
                ),
            )
            return

        connection.execute(
            """
            UPDATE strategies
            SET status = ?, updated_at = ?
            WHERE status = ? AND task_type = ? AND id != ?
            """,
            (
                StrategyStatus.RETIRED.value,
                updated_at,
                StrategyStatus.CHAMPION.value,
                task_type,
                excluding_strategy_id,
            ),
        )


def _strategy_to_row(strategy: Strategy) -> tuple[Any, ...]:
    """Convert a Strategy into a SQLite row tuple."""
    return (
        strategy.id,
        strategy.version,
        strategy.name,
        strategy.task_type,
        strategy.status.value,
        strategy.genome.model_dump_json(),
        json.dumps(strategy.parent_ids),
        strategy.created_at.isoformat(),
        strategy.updated_at.isoformat(),
        strategy.notes,
    )


def _row_to_strategy(row: sqlite3.Row) -> Strategy:
    """Convert a SQLite row into a Strategy model."""
    return Strategy(
        id=row["id"],
        version=row["version"],
        name=row["name"],
        task_type=row["task_type"],
        status=row["status"],
        genome=json.loads(row["genome_json"]),
        parent_ids=json.loads(row["parent_ids_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        notes=row["notes"],
    )


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _next_updated_at_iso(previous_timestamp: str) -> str:
    """Return a timestamp later than the previous updated_at value."""
    previous = datetime.fromisoformat(previous_timestamp)
    current = datetime.now(UTC)
    if current <= previous:
        current = previous + timedelta(microseconds=1)
    return current.isoformat()
