"""Promotion state controller and audit events."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.promotion.gate import PromotionGateResult
from goagentx.registry.db import initialize_database


class PromotionControllerError(RuntimeError):
    """Raised when a promotion transition is not allowed."""


class PromotionRegistry(Protocol):
    """Strategy registry surface needed by the promotion controller."""

    database_path: Path

    def get(self, strategy_id: str) -> Strategy:
        """Return a Strategy by id."""

    def update_status(self, strategy_id: str, status: str | StrategyStatus) -> None:
        """Update one Strategy status."""


class StrictModel(BaseModel):
    """Base model that rejects unknown promotion-controller fields."""

    model_config = ConfigDict(extra="forbid")


class PromotionEvent(StrictModel):
    """Audited strategy promotion status change."""

    id: str = Field(..., min_length=1)
    strategy_id: str = Field(..., min_length=1)
    from_status: StrategyStatus
    to_status: StrategyStatus
    reason: str = Field(..., min_length=1)
    experiment_id: str | None = None
    created_at: datetime


class PromotionResult(StrictModel):
    """Promotion transition result."""

    strategy: Strategy
    event: PromotionEvent
    gate: PromotionGateResult


class PromotionController:
    """Advance candidate strategies through audited promotion states."""

    def __init__(
        self,
        registry: PromotionRegistry,
        *,
        database_path: str | Path | None = None,
    ) -> None:
        """Create a controller backed by the registry database."""
        self.registry = registry
        resolved_database_path = database_path or registry.database_path
        self.database_path = initialize_database(resolved_database_path)

    def promote(
        self,
        strategy_id: str,
        *,
        target_status: StrategyStatus | str,
        gate: PromotionGateResult,
        reason: str | None = None,
    ) -> PromotionResult:
        """Promote one strategy to the next lifecycle state."""
        target = StrategyStatus(target_status)
        current = self.registry.get(strategy_id)
        _validate_transition(current, target=target, gate=gate)

        resolved_reason = reason or f"promotion_gate:{gate.decision.value}"
        self.registry.update_status(strategy_id, target)
        promoted = self.registry.get(strategy_id)
        event = self._save_event(
            strategy_id=strategy_id,
            from_status=current.status,
            to_status=target,
            reason=resolved_reason,
            experiment_id=gate.metrics.experiment_id,
        )
        return PromotionResult(strategy=promoted, event=event, gate=gate)

    def list_events(self, *, strategy_id: str | None = None) -> list[PromotionEvent]:
        """List promotion events, oldest first."""
        where_clause = ""
        parameters: tuple[str, ...] = ()
        if strategy_id is not None:
            where_clause = " WHERE strategy_id = ?"
            parameters = (strategy_id,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, strategy_id, from_status, to_status, reason,
                       experiment_id, created_at
                FROM promotion_events
                {where_clause}
                ORDER BY created_at ASC, id ASC
                """,
                parameters,
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def _save_event(
        self,
        *,
        strategy_id: str,
        from_status: StrategyStatus,
        to_status: StrategyStatus,
        reason: str,
        experiment_id: str | None,
    ) -> PromotionEvent:
        created_at = datetime.now(UTC)
        event = PromotionEvent(
            id=_event_id(
                strategy_id=strategy_id,
                from_status=from_status,
                to_status=to_status,
                created_at=created_at,
            ),
            strategy_id=strategy_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            experiment_id=experiment_id,
            created_at=created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO promotion_events (
                  id, strategy_id, from_status, to_status, reason,
                  experiment_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                _event_to_row(event),
            )
        return event

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _validate_transition(
    strategy: Strategy,
    *,
    target: StrategyStatus,
    gate: PromotionGateResult,
) -> None:
    if gate.metrics.candidate_id != strategy.id:
        raise PromotionControllerError("gate result does not match strategy id")
    if not gate.approved:
        raise PromotionControllerError("promotion gate rejected candidate")
    allowed_targets = _allowed_targets(strategy.status)
    if target not in allowed_targets:
        raise PromotionControllerError(
            f"cannot promote {strategy.status.value} strategy to {target.value}"
        )


def _allowed_targets(status: StrategyStatus) -> set[StrategyStatus]:
    if status is StrategyStatus.CANDIDATE:
        return {StrategyStatus.SHADOW}
    if status is StrategyStatus.SHADOW:
        return {StrategyStatus.CANARY}
    if status is StrategyStatus.CANARY:
        return {StrategyStatus.CHAMPION}
    return set()


def _event_id(
    *,
    strategy_id: str,
    from_status: StrategyStatus,
    to_status: StrategyStatus,
    created_at: datetime,
) -> str:
    timestamp = created_at.strftime("%Y%m%d%H%M%S%f")
    return (
        f"promotion-{strategy_id}-"
        f"{from_status.value}-to-{to_status.value}-{timestamp}"
    )


def _event_to_row(event: PromotionEvent) -> tuple[object, ...]:
    return (
        event.id,
        event.strategy_id,
        event.from_status.value,
        event.to_status.value,
        event.reason,
        event.experiment_id,
        event.created_at.isoformat(),
    )


def _row_to_event(row: sqlite3.Row) -> PromotionEvent:
    return PromotionEvent(
        id=row["id"],
        strategy_id=row["strategy_id"],
        from_status=row["from_status"],
        to_status=row["to_status"],
        reason=row["reason"],
        experiment_id=row["experiment_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
