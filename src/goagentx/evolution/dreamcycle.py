"""DreamCycle orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.arena.runner import (
    QuickRejectResult,
    QuickRejectSettings,
    run_quick_reject_from_settings,
)
from goagentx.core.run import AgentRunner
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task, TaskRun
from goagentx.evolution.mutation import MutationKind, StrategyMutator
from goagentx.evolution.scheduler import (
    DegradationDetector,
    DegradationResult,
    DegradationSettings,
)


class DreamCycleError(RuntimeError):
    """Raised when DreamCycle cannot be run."""


class StrategyRegistryProtocol(Protocol):
    """Strategy registry surface needed by DreamCycle."""

    def get(self, strategy_id: str) -> Strategy:
        """Return a Strategy by id."""

    def create(self, strategy: Strategy) -> Strategy:
        """Persist a new Strategy."""


class DreamCycleTaskStore(Protocol):
    """Task and TaskRun store surface needed by DreamCycle."""

    def list_tasks(
        self,
        *,
        task_type: str | None = None,
        bucket: str | None = None,
        task_set_id: str | None = None,
        limit: int | None = None,
    ) -> list[Task]:
        """List persisted tasks."""

    def list_recent_runs(
        self,
        *,
        limit: int = 10,
        task_id: str | None = None,
        strategy_id: str | None = None,
        experiment_id: str | None = None,
        task_type: str | None = None,
    ) -> list[TaskRun]:
        """List recent TaskRuns."""

    def save_run(self, task_run: TaskRun) -> TaskRun:
        """Persist one TaskRun."""


class StrictModel(BaseModel):
    """Base model that rejects unknown DreamCycle fields."""

    model_config = ConfigDict(extra="forbid")


class DreamCycleCandidateResult(StrictModel):
    """A generated candidate and its optional Arena result."""

    candidate: Strategy
    mutation_kind: MutationKind
    quick_reject: QuickRejectResult | None = None


class DreamCycleResult(StrictModel):
    """DreamCycle orchestration result."""

    champion_id: str
    task_type: str | None = None
    triggered: bool
    reason: str
    detection: DegradationResult
    candidates: list[DreamCycleCandidateResult] = Field(default_factory=list)
    auto_run_arena: bool
    audit_log_path: Path


def run_dreamcycle(
    *,
    champion_id: str,
    registry: StrategyRegistryProtocol,
    task_store: DreamCycleTaskStore,
    mutator: StrategyMutator,
    scorer: Scorer,
    runner: AgentRunner,
    evolution_settings: DegradationSettings,
    arena_settings: QuickRejectSettings,
    audit_log_path: str | Path,
    task_type: str | None = None,
    candidate_count: int = 3,
    mutation_kinds: Sequence[MutationKind | str] | None = None,
    auto_run_arena: bool = True,
    manual_trigger: bool = False,
    seed: int | None = 0,
    candidate_runner: AgentRunner | None = None,
) -> DreamCycleResult:
    """Run DreamCycle when degradation is detected or manual trigger is set."""
    if not 1 <= candidate_count <= 3:
        raise DreamCycleError("candidate_count must be between 1 and 3")

    audit_logger = DreamCycleAuditLogger(audit_log_path)
    champion = registry.get(champion_id)
    resolved_task_type = task_type if task_type is not None else champion.task_type

    audit_logger.write(
        "dreamcycle_started",
        {
            "champion_id": champion.id,
            "task_type": resolved_task_type,
            "candidate_count": candidate_count,
            "auto_run_arena": auto_run_arena,
            "manual_trigger": manual_trigger,
        },
    )
    detection = DegradationDetector(task_store, evolution_settings).detect(
        strategy_id=champion.id,
        task_type=resolved_task_type,
    )
    audit_logger.write("degradation_checked", detection)

    should_generate = detection.triggered or manual_trigger
    if not should_generate:
        audit_logger.write(
            "dreamcycle_skipped",
            {
                "champion_id": champion.id,
                "reason": detection.reason,
            },
        )
        return DreamCycleResult(
            champion_id=champion.id,
            task_type=resolved_task_type,
            triggered=False,
            reason=detection.reason,
            detection=detection,
            candidates=[],
            auto_run_arena=auto_run_arena,
            audit_log_path=audit_logger.path,
        )

    arena_tasks = _tasks_for_arena(
        task_store,
        task_type=resolved_task_type,
        auto_run_arena=auto_run_arena,
    )
    selected_kinds = _selected_mutation_kinds(mutation_kinds, candidate_count)
    candidate_results: list[DreamCycleCandidateResult] = []
    for index, mutation_kind in enumerate(selected_kinds, start=1):
        candidate = mutator.mutate(
            champion,
            kind=mutation_kind,
            candidate_id=_candidate_id(
                champion_id=champion.id,
                mutation_kind=mutation_kind,
                index=index,
            ),
        )
        created_candidate = registry.create(candidate)
        audit_logger.write(
            "candidate_created",
            {
                "candidate_id": created_candidate.id,
                "parent_id": champion.id,
                "mutation_kind": mutation_kind.value,
            },
        )

        quick_reject_result: QuickRejectResult | None = None
        if auto_run_arena:
            quick_reject_result = run_quick_reject_from_settings(
                champion=champion,
                candidate=created_candidate,
                tasks=arena_tasks,
                champion_runner=runner,
                candidate_runner=candidate_runner,
                scorer=scorer,
                settings=arena_settings,
                seed=seed,
                experiment_id=f"dreamcycle-{champion.id}-vs-{created_candidate.id}",
                task_store=task_store,
            )
            audit_logger.write(
                "quick_reject_completed",
                {
                    "candidate_id": created_candidate.id,
                    "decision": quick_reject_result.decision.value,
                    "failed_checks": quick_reject_result.failed_checks,
                },
            )

        candidate_results.append(
            DreamCycleCandidateResult(
                candidate=created_candidate,
                mutation_kind=mutation_kind,
                quick_reject=quick_reject_result,
            )
        )

    audit_logger.write(
        "dreamcycle_completed",
        {
            "champion_id": champion.id,
            "generated_candidates": [item.candidate.id for item in candidate_results],
        },
    )
    return DreamCycleResult(
        champion_id=champion.id,
        task_type=resolved_task_type,
        triggered=True,
        reason="manual_trigger" if manual_trigger else detection.reason,
        detection=detection,
        candidates=candidate_results,
        auto_run_arena=auto_run_arena,
        audit_log_path=audit_logger.path,
    )


class DreamCycleAuditLogger:
    """Append DreamCycle audit events as JSONL."""

    def __init__(self, path: str | Path) -> None:
        """Create an audit logger for a JSONL file."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: BaseModel | dict[str, Any]) -> None:
        """Append one audit event."""
        event = {
            "event_type": event_type,
            "created_at": datetime.now(UTC).isoformat(),
            "payload": _jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")


def _tasks_for_arena(
    task_store: DreamCycleTaskStore,
    *,
    task_type: str | None,
    auto_run_arena: bool,
) -> list[Task]:
    """Return tasks needed by Quick Reject."""
    if not auto_run_arena:
        return []
    tasks = task_store.list_tasks(task_type=task_type)
    if not tasks:
        raise DreamCycleError("auto_run_arena requires at least one matching task")
    return tasks


def _selected_mutation_kinds(
    mutation_kinds: Sequence[MutationKind | str] | None,
    candidate_count: int,
) -> list[MutationKind]:
    """Resolve mutation kinds for candidate generation."""
    default_kinds = [MutationKind.PARAMETER, MutationKind.PROMPT, MutationKind.TOOL]
    source_kinds = mutation_kinds or default_kinds
    resolved_kinds = [MutationKind(kind) for kind in source_kinds]
    if not resolved_kinds:
        raise DreamCycleError("at least one mutation kind is required")
    return [resolved_kinds[index % len(resolved_kinds)] for index in range(candidate_count)]


def _candidate_id(
    *,
    champion_id: str,
    mutation_kind: MutationKind,
    index: int,
) -> str:
    """Build a deterministic candidate id for one DreamCycle run."""
    return f"{champion_id}-dream-{mutation_kind.value}-{index:02d}"


def _jsonable(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic models and dicts into JSON-friendly mappings."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
