"""SQLite-backed evaluation experiment store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from goagentx.registry.db import initialize_database


class EvalExperimentStoreError(RuntimeError):
    """Base error for evaluation experiment store failures."""


class EvalExperimentNotFoundError(EvalExperimentStoreError):
    """Raised when a requested evaluation experiment does not exist."""


class StrictModel(BaseModel):
    """Base model that rejects unknown experiment fields."""

    model_config = ConfigDict(extra="forbid")


class EvalExperiment(StrictModel):
    """Persisted Arena evaluation experiment summary."""

    id: str = Field(..., min_length=1)
    champion_id: str = Field(..., min_length=1)
    candidate_id: str = Field(..., min_length=1)
    task_set_id: str = Field(..., min_length=1)
    quick_reject_passed: bool
    win_rate: float = Field(..., ge=0.0, le=1.0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_score_delta: float
    cost_delta: float
    latency_delta: float
    safety_violation_count: int = Field(default=0, ge=0)
    critical_bucket_regression: bool = False
    verdict: str = Field(..., min_length=1)
    report_path: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvalExperimentStore:
    """Persist Arena evaluation summaries in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        """Initialize the store and ensure the SQLite schema exists."""
        self.database_path = initialize_database(database_path)

    def save(self, experiment: EvalExperiment) -> EvalExperiment:
        """Save an evaluation experiment summary."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO eval_experiments (
                  id, champion_id, candidate_id, task_set_id, quick_reject_passed,
                  win_rate, p_value, avg_score_delta, cost_delta, latency_delta,
                  safety_violation_count, critical_bucket_regression, verdict,
                  report_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _experiment_to_row(experiment),
            )
        return self.get(experiment.id)

    def get(self, experiment_id: str) -> EvalExperiment:
        """Return an evaluation experiment by id."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, champion_id, candidate_id, task_set_id, quick_reject_passed,
                       win_rate, p_value, avg_score_delta, cost_delta, latency_delta,
                       safety_violation_count, critical_bucket_regression, verdict,
                       report_path, created_at
                FROM eval_experiments
                WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise EvalExperimentNotFoundError(
                f"Evaluation experiment not found: {experiment_id}"
            )
        return _row_to_experiment(row)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection configured for dict-like rows."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _experiment_to_row(experiment: EvalExperiment) -> tuple[object, ...]:
    """Convert an EvalExperiment into a SQLite row tuple."""
    return (
        experiment.id,
        experiment.champion_id,
        experiment.candidate_id,
        experiment.task_set_id,
        int(experiment.quick_reject_passed),
        experiment.win_rate,
        experiment.p_value,
        experiment.avg_score_delta,
        experiment.cost_delta,
        experiment.latency_delta,
        experiment.safety_violation_count,
        int(experiment.critical_bucket_regression),
        experiment.verdict,
        str(experiment.report_path) if experiment.report_path is not None else None,
        experiment.created_at.isoformat(),
    )


def _row_to_experiment(row: sqlite3.Row) -> EvalExperiment:
    """Convert a SQLite row into an EvalExperiment model."""
    return EvalExperiment(
        id=row["id"],
        champion_id=row["champion_id"],
        candidate_id=row["candidate_id"],
        task_set_id=row["task_set_id"],
        quick_reject_passed=bool(row["quick_reject_passed"]),
        win_rate=row["win_rate"],
        p_value=row["p_value"],
        avg_score_delta=row["avg_score_delta"],
        cost_delta=row["cost_delta"],
        latency_delta=row["latency_delta"],
        safety_violation_count=row["safety_violation_count"],
        critical_bucket_regression=bool(row["critical_bucket_regression"]),
        verdict=row["verdict"],
        report_path=Path(row["report_path"]) if row["report_path"] else None,
        created_at=datetime.fromisoformat(row["created_at"]),
    )
