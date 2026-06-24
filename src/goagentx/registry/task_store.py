"""SQLite-backed task and task-run store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from goagentx.core.task import Task, TaskRun, TaskSet
from goagentx.registry.db import initialize_database


class TaskStoreError(RuntimeError):
    """Base error for task store failures."""


class TaskStoreNotFoundError(TaskStoreError):
    """Raised when a requested task does not exist."""


class TaskStore:
    """Persist tasks and task runs in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        """Initialize the store and ensure the SQLite schema exists."""
        self.database_path = initialize_database(database_path)

    def save_task(self, task: Task, *, task_set_id: str | None = None) -> Task:
        """Save a task, replacing any existing task with the same id."""
        stored_task = _task_with_task_set_id(task, task_set_id or task.task_set_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO tasks (
                  id, task_set_id, task_type, bucket, input_json, expected_json,
                  tags_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _task_to_row(stored_task),
            )
        return self.get_task(stored_task.id)

    def save_task_set(self, task_set: TaskSet) -> list[Task]:
        """Save every task in a task set with the set id attached."""
        return [self.save_task(task, task_set_id=task_set.id) for task in task_set.tasks]

    def get_task(self, task_id: str) -> Task:
        """Return a task by id."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, task_set_id, task_type, bucket, input_json, expected_json,
                       tags_json, created_at
                FROM tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            raise TaskStoreNotFoundError(f"Task not found: {task_id}")
        return _row_to_task(row)

    def list_tasks(
        self,
        *,
        task_type: str | None = None,
        bucket: str | None = None,
        task_set_id: str | None = None,
        limit: int | None = None,
    ) -> list[Task]:
        """List tasks with optional filters."""
        where_clauses, parameters = _task_filter_clauses(
            task_type=task_type,
            bucket=bucket,
            task_set_id=task_set_id,
        )
        sql = """
            SELECT id, task_set_id, task_type, bucket, input_json, expected_json,
                   tags_json, created_at
            FROM tasks
        """
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY created_at ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
        return [_row_to_task(row) for row in rows]

    def sample_tasks(
        self,
        *,
        task_type: str | None = None,
        bucket: str | None = None,
        task_set_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 5,
    ) -> list[Task]:
        """Return a random sample of tasks matching filters and time window."""
        where_clauses, parameters = _task_filter_clauses(
            task_type=task_type,
            bucket=bucket,
            task_set_id=task_set_id,
        )
        if created_after is not None:
            where_clauses.append("created_at >= ?")
            parameters.append(created_after.isoformat())
        if created_before is not None:
            where_clauses.append("created_at <= ?")
            parameters.append(created_before.isoformat())

        sql = """
            SELECT id, task_set_id, task_type, bucket, input_json, expected_json,
                   tags_json, created_at
            FROM tasks
        """
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY RANDOM() LIMIT ?"
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
        return [_row_to_task(row) for row in rows]

    def save_run(self, task_run: TaskRun) -> TaskRun:
        """Save a task run, replacing any existing run with the same id."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO task_runs (
                  id, task_id, strategy_id, experiment_id, output_json, score,
                  score_breakdown_json, success, cost, latency_ms, token_count,
                  tool_calls_json, error_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _task_run_to_row(task_run),
            )
        return task_run

    def list_recent_runs(
        self,
        *,
        limit: int = 10,
        task_id: str | None = None,
        strategy_id: str | None = None,
        experiment_id: str | None = None,
    ) -> list[TaskRun]:
        """List recent task runs with optional filters."""
        where_clauses: list[str] = []
        parameters: list[Any] = []
        if task_id is not None:
            where_clauses.append("task_id = ?")
            parameters.append(task_id)
        if strategy_id is not None:
            where_clauses.append("strategy_id = ?")
            parameters.append(strategy_id)
        if experiment_id is not None:
            where_clauses.append("experiment_id = ?")
            parameters.append(experiment_id)

        sql = """
            SELECT id, task_id, strategy_id, experiment_id, output_json, score,
                   score_breakdown_json, success, cost, latency_ms, token_count,
                   tool_calls_json, error_json, created_at
            FROM task_runs
        """
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
        return [_row_to_task_run(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured for dict-like rows."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _task_filter_clauses(
    *,
    task_type: str | None,
    bucket: str | None,
    task_set_id: str | None,
) -> tuple[list[str], list[Any]]:
    """Build SQL filters for task queries."""
    where_clauses: list[str] = []
    parameters: list[Any] = []
    if task_type is not None:
        where_clauses.append("task_type = ?")
        parameters.append(task_type)
    if bucket is not None:
        where_clauses.append("bucket = ?")
        parameters.append(bucket)
    if task_set_id is not None:
        where_clauses.append("task_set_id = ?")
        parameters.append(task_set_id)
    return where_clauses, parameters


def _task_with_task_set_id(task: Task, task_set_id: str | None) -> Task:
    """Return a task copy with task_set_id set."""
    data = task.model_dump(mode="python")
    data["task_set_id"] = task_set_id
    return Task.model_validate(data)


def _task_to_row(task: Task) -> tuple[Any, ...]:
    """Convert a Task to a SQLite row tuple."""
    return (
        task.id,
        task.task_set_id,
        task.task_type,
        task.bucket,
        json.dumps(task.input_json),
        json.dumps(task.expected_json) if task.expected_json is not None else None,
        json.dumps(task.tags),
        task.created_at.isoformat(),
    )


def _row_to_task(row: sqlite3.Row) -> Task:
    """Convert a SQLite row into a Task model."""
    return Task(
        id=row["id"],
        task_set_id=row["task_set_id"],
        task_type=row["task_type"],
        bucket=row["bucket"],
        input_json=json.loads(row["input_json"]),
        expected_json=json.loads(row["expected_json"]) if row["expected_json"] else None,
        tags=json.loads(row["tags_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _task_run_to_row(task_run: TaskRun) -> tuple[Any, ...]:
    """Convert a TaskRun to a SQLite row tuple."""
    return (
        task_run.id,
        task_run.task_id,
        task_run.strategy_id,
        task_run.experiment_id,
        json.dumps(task_run.output_json),
        task_run.score,
        json.dumps(task_run.score_breakdown),
        int(task_run.success),
        task_run.cost,
        task_run.latency_ms,
        task_run.token_count,
        json.dumps(task_run.tool_calls),
        json.dumps(task_run.error_json) if task_run.error_json is not None else None,
        task_run.created_at.isoformat(),
    )


def _row_to_task_run(row: sqlite3.Row) -> TaskRun:
    """Convert a SQLite row into a TaskRun model."""
    return TaskRun(
        id=row["id"],
        task_id=row["task_id"],
        strategy_id=row["strategy_id"],
        experiment_id=row["experiment_id"],
        output_json=json.loads(row["output_json"]),
        score=row["score"],
        score_breakdown=json.loads(row["score_breakdown_json"]),
        success=bool(row["success"]),
        cost=row["cost"],
        latency_ms=row["latency_ms"],
        token_count=row["token_count"],
        tool_calls=json.loads(row["tool_calls_json"]),
        error_json=json.loads(row["error_json"]) if row["error_json"] else None,
        created_at=datetime.fromisoformat(row["created_at"]),
    )
