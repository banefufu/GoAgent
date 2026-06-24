"""Rollback controls for promoted strategies."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.promotion.controller import PromotionEvent
from goagentx.registry.db import initialize_database
from goagentx.registry.strategy_registry import StrategyNotFoundError


class RollbackControllerError(RuntimeError):
    """Raised when a rollback cannot be performed safely."""


class RollbackRegistry(Protocol):
    """Strategy registry surface needed by the rollback controller."""

    database_path: Path

    def get(self, strategy_id: str) -> Strategy:
        """Return a Strategy by id."""

    def get_champion(self, task_type: str | None = None) -> Strategy:
        """Return the current champion for a task type."""

    def update_status(self, strategy_id: str, status: str | StrategyStatus) -> None:
        """Update one Strategy status."""


class StrictModel(BaseModel):
    """Base model that rejects unknown rollback-controller fields."""

    model_config = ConfigDict(extra="forbid")


class RollbackResult(StrictModel):
    """Rollback result with restored and failed strategy details."""

    restored_strategy: Strategy
    failed_strategy: Strategy | None = None
    events: list[PromotionEvent] = Field(default_factory=list)


class RollbackController:
    """Restore a previous champion or roll back a failed shadow/canary."""

    def __init__(
        self,
        registry: RollbackRegistry,
        *,
        database_path: str | Path | None = None,
    ) -> None:
        """Create a rollback controller backed by the registry database."""
        self.registry = registry
        resolved_database_path = database_path or registry.database_path
        self.database_path = initialize_database(resolved_database_path)

    def rollback(
        self,
        to_strategy_id: str,
        *,
        failed_strategy_id: str | None = None,
        failed_status: StrategyStatus | str = StrategyStatus.ROLLED_BACK,
        reason: str | None = None,
    ) -> RollbackResult:
        """Rollback to a stable strategy and audit any status changes."""
        target = self.registry.get(to_strategy_id)
        _validate_target(target)

        resolved_failed_status = StrategyStatus(failed_status)
        _validate_failed_status(resolved_failed_status)

        failed_before = self._resolve_failed_strategy(
            target,
            failed_strategy_id=failed_strategy_id,
        )
        if failed_before is not None:
            _validate_failed_strategy(failed_before, target=target)

        if target.status is StrategyStatus.CHAMPION and failed_before is None:
            raise RollbackControllerError(
                "target strategy is already champion; provide failed_strategy_id "
                "to roll back a shadow or canary"
            )

        resolved_reason = reason or f"rollback_to:{target.id}"
        events: list[PromotionEvent] = []
        failed_after: Strategy | None = None

        if failed_before is not None:
            self.registry.update_status(failed_before.id, resolved_failed_status)
            failed_after = self.registry.get(failed_before.id)
            events.append(
                self._save_event(
                    strategy_id=failed_before.id,
                    from_status=failed_before.status,
                    to_status=resolved_failed_status,
                    reason=resolved_reason,
                )
            )

        if target.status is not StrategyStatus.CHAMPION:
            self.registry.update_status(target.id, StrategyStatus.CHAMPION)
            restored = self.registry.get(target.id)
            events.append(
                self._save_event(
                    strategy_id=target.id,
                    from_status=target.status,
                    to_status=StrategyStatus.CHAMPION,
                    reason=resolved_reason,
                )
            )
        else:
            restored = self.registry.get(target.id)

        return RollbackResult(
            restored_strategy=restored,
            failed_strategy=failed_after,
            events=events,
        )

    def _resolve_failed_strategy(
        self,
        target: Strategy,
        *,
        failed_strategy_id: str | None,
    ) -> Strategy | None:
        if failed_strategy_id is not None:
            failed = self.registry.get(failed_strategy_id)
            if target.status is StrategyStatus.RETIRED:
                current_champion = self._get_current_champion(target)
                if failed.id != current_champion.id:
                    raise RollbackControllerError(
                        "failed strategy must be the current champion when "
                        "restoring a retired target"
                    )
            return failed
        if target.status is StrategyStatus.CHAMPION:
            return None
        current_champion = self._get_current_champion(target)
        if current_champion.id == target.id:
            return None
        return current_champion

    def _get_current_champion(self, target: Strategy) -> Strategy:
        try:
            return self.registry.get_champion(target.task_type)
        except StrategyNotFoundError as exc:
            raise RollbackControllerError(
                "cannot rollback without an active champion in the target domain"
            ) from exc

    def _save_event(
        self,
        *,
        strategy_id: str,
        from_status: StrategyStatus,
        to_status: StrategyStatus,
        reason: str,
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
            experiment_id=None,
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


def _validate_target(target: Strategy) -> None:
    if target.status not in {StrategyStatus.CHAMPION, StrategyStatus.RETIRED}:
        raise RollbackControllerError(
            "rollback target must be a stable champion or retired champion"
        )


def _validate_failed_status(status: StrategyStatus) -> None:
    if status not in {StrategyStatus.ROLLED_BACK, StrategyStatus.RETIRED}:
        raise RollbackControllerError(
            "failed strategy status must be rolled_back or retired"
        )


def _validate_failed_strategy(failed: Strategy, *, target: Strategy) -> None:
    if failed.id == target.id:
        raise RollbackControllerError("failed strategy cannot be the rollback target")
    if failed.task_type != target.task_type:
        raise RollbackControllerError(
            "failed strategy must be in the same task_type domain as rollback target"
        )
    if failed.status not in {
        StrategyStatus.SHADOW,
        StrategyStatus.CANARY,
        StrategyStatus.CHAMPION,
    }:
        raise RollbackControllerError(
            "failed strategy must be shadow, canary, or champion"
        )


def _event_id(
    *,
    strategy_id: str,
    from_status: StrategyStatus,
    to_status: StrategyStatus,
    created_at: datetime,
) -> str:
    timestamp = created_at.strftime("%Y%m%d%H%M%S%f")
    return (
        f"rollback-{strategy_id}-"
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
