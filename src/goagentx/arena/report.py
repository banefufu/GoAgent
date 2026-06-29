"""Full Arena evaluation and Markdown report generation."""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from goagentx.arena.paired_eval import (
    PairOutcome,
    PairedEvaluationResult,
    TaskRunStore,
    evaluate_pair,
)
from goagentx.arena.runner import select_quick_reject_tasks
from goagentx.arena.stats import (
    PermutationAlternative,
    SignificanceResult,
    permutation_test_paired_result,
)
from goagentx.core.run import AgentRunner
from goagentx.core.scoring import Scorer
from goagentx.core.strategy import Strategy
from goagentx.core.task import Task
from goagentx.registry.experiment_store import EvalExperiment, EvalExperimentStore


class FullEvalError(RuntimeError):
    """Raised when full evaluation cannot be run."""


class FullEvalVerdict(StrEnum):
    """Full Eval decisions."""

    PROMOTE_READY = "promote_ready"
    REJECT = "reject"


class StrictModel(BaseModel):
    """Base model that rejects unknown full-eval fields."""

    model_config = ConfigDict(extra="forbid")


class FullEvaluationResult(StrictModel):
    """Full Eval result with report and persistence metadata."""

    experiment_id: str
    task_set_id: str
    verdict: FullEvalVerdict
    full_eval_passed: bool
    failed_checks: list[str]
    selected_task_ids: list[str] = Field(..., min_length=1)
    evaluation: PairedEvaluationResult
    significance: SignificanceResult
    cost_delta: float
    latency_delta: float
    safety_violation_count: int = Field(..., ge=0)
    critical_bucket_regression: bool
    report_path: Path | None = None


class FullEvalSettings(Protocol):
    """Arena settings surface needed by Full Eval."""

    full_eval_rounds: int
    min_win_rate: float
    p_value_threshold: float


class FullEvalGateSettings(Protocol):
    """Promotion gate settings surface needed by Full Eval."""

    min_score_delta: float
    max_cost_delta: float
    max_latency_delta: float
    require_no_safety_violation: bool
    require_no_critical_bucket_regression: bool


def run_full_eval(
    *,
    champion: Strategy,
    candidate: Strategy,
    tasks: Sequence[Task],
    champion_runner: AgentRunner,
    scorer: Scorer,
    candidate_runner: AgentRunner | None = None,
    rounds: int = 50,
    task_set_id: str = "ad_hoc",
    min_win_rate: float = 0.55,
    max_p_value: float = 0.05,
    min_score_delta: float = 0.0,
    max_cost_delta: float = 0.20,
    max_latency_delta: float = 0.20,
    require_no_safety_violation: bool = True,
    require_no_critical_bucket_regression: bool = True,
    tie_threshold: float = 0.0,
    seed: int | None = 0,
    experiment_id: str | None = None,
    report_directory: str | Path | None = None,
    experiment_store: EvalExperimentStore | None = None,
    task_store: TaskRunStore | None = None,
) -> FullEvaluationResult:
    """Run Full Eval and optionally write a report and experiment row."""
    _validate_full_eval_config(
        rounds=rounds,
        min_win_rate=min_win_rate,
        max_p_value=max_p_value,
        max_cost_delta=max_cost_delta,
        max_latency_delta=max_latency_delta,
        tie_threshold=tie_threshold,
    )
    selected_tasks = select_full_eval_tasks(tasks, rounds=rounds, seed=seed)
    resolved_experiment_id = experiment_id or make_full_eval_experiment_id(
        champion_strategy_id=champion.id,
        candidate_strategy_id=candidate.id,
    )
    evaluation = evaluate_pair(
        champion=champion,
        candidate=candidate,
        tasks=selected_tasks,
        champion_runner=champion_runner,
        candidate_runner=candidate_runner,
        scorer=scorer,
        tie_threshold=tie_threshold,
        experiment_id=resolved_experiment_id,
        task_store=task_store,
    )
    significance = permutation_test_paired_result(
        evaluation,
        alternative=PermutationAlternative.GREATER,
        alpha=max_p_value,
        seed=seed,
    )
    cost_delta = _relative_delta(
        _average([result.champion_run.cost for result in evaluation.results]),
        _average([result.candidate_run.cost for result in evaluation.results]),
    )
    latency_delta = _relative_delta(
        _average([result.champion_run.latency_ms for result in evaluation.results]),
        _average([result.candidate_run.latency_ms for result in evaluation.results]),
    )
    safety_violation_count = _candidate_safety_violation_count(evaluation)
    critical_bucket_regression = _has_critical_bucket_regression(
        evaluation,
        tie_threshold=tie_threshold,
    )
    failed_checks = _full_eval_failed_checks(
        evaluation=evaluation,
        significance=significance,
        cost_delta=cost_delta,
        latency_delta=latency_delta,
        safety_violation_count=safety_violation_count,
        critical_bucket_regression=critical_bucket_regression,
        min_win_rate=min_win_rate,
        min_score_delta=min_score_delta,
        max_cost_delta=max_cost_delta,
        max_latency_delta=max_latency_delta,
        require_no_safety_violation=require_no_safety_violation,
        require_no_critical_bucket_regression=require_no_critical_bucket_regression,
    )
    verdict = FullEvalVerdict.REJECT if failed_checks else FullEvalVerdict.PROMOTE_READY
    result = FullEvaluationResult(
        experiment_id=resolved_experiment_id,
        task_set_id=task_set_id,
        verdict=verdict,
        full_eval_passed=verdict is FullEvalVerdict.PROMOTE_READY,
        failed_checks=failed_checks,
        selected_task_ids=[task.id for task in selected_tasks],
        evaluation=evaluation,
        significance=significance,
        cost_delta=cost_delta,
        latency_delta=latency_delta,
        safety_violation_count=safety_violation_count,
        critical_bucket_regression=critical_bucket_regression,
    )
    if report_directory is not None:
        result = write_full_eval_report(result, report_directory)
    if experiment_store is not None:
        experiment_store.save(full_eval_result_to_experiment(result))
    return result


def run_full_eval_from_settings(
    *,
    champion: Strategy,
    candidate: Strategy,
    tasks: Sequence[Task],
    champion_runner: AgentRunner,
    scorer: Scorer,
    arena_settings: FullEvalSettings,
    gate_settings: FullEvalGateSettings,
    candidate_runner: AgentRunner | None = None,
    task_set_id: str = "ad_hoc",
    tie_threshold: float = 0.0,
    seed: int | None = 0,
    experiment_id: str | None = None,
    report_directory: str | Path | None = None,
    experiment_store: EvalExperimentStore | None = None,
    task_store: TaskRunStore | None = None,
) -> FullEvaluationResult:
    """Run Full Eval using validated arena and promotion settings."""
    return run_full_eval(
        champion=champion,
        candidate=candidate,
        tasks=tasks,
        champion_runner=champion_runner,
        candidate_runner=candidate_runner,
        scorer=scorer,
        rounds=arena_settings.full_eval_rounds,
        task_set_id=task_set_id,
        min_win_rate=arena_settings.min_win_rate,
        max_p_value=arena_settings.p_value_threshold,
        min_score_delta=gate_settings.min_score_delta,
        max_cost_delta=gate_settings.max_cost_delta,
        max_latency_delta=gate_settings.max_latency_delta,
        require_no_safety_violation=gate_settings.require_no_safety_violation,
        require_no_critical_bucket_regression=(
            gate_settings.require_no_critical_bucket_regression
        ),
        tie_threshold=tie_threshold,
        seed=seed,
        experiment_id=experiment_id,
        report_directory=report_directory,
        experiment_store=experiment_store,
        task_store=task_store,
    )


def select_full_eval_tasks(
    tasks: Sequence[Task],
    *,
    rounds: int,
    seed: int | None = 0,
) -> list[Task]:
    """Select a deterministic stratified Full Eval task sample."""
    return select_quick_reject_tasks(tasks, rounds=rounds, seed=seed)


def render_full_eval_report(result: FullEvaluationResult) -> str:
    """Render a Full Eval result as Markdown."""
    lines = [
        "# Arena Full Eval Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| experiment_id | `{result.experiment_id}` |",
        f"| verdict | `{result.verdict.value}` |",
        f"| win_rate | {_format_float(result.evaluation.win_rate)} |",
        f"| p_value | {_format_optional_float(result.significance.p_value)} |",
        f"| avg_score_delta | {_format_float(result.evaluation.avg_score_delta)} |",
        f"| cost_delta | {_format_percent(result.cost_delta)} |",
        f"| latency_delta | {_format_percent(result.latency_delta)} |",
        f"| safety_violation_count | {result.safety_violation_count} |",
        f"| critical_bucket_regression | {str(result.critical_bucket_regression).lower()} |",
        "",
        "## Verdict Reasons",
        "",
    ]
    if result.failed_checks:
        lines.extend(f"- {check}" for check in result.failed_checks)
    else:
        lines.append("- passed_all_full_eval_checks")

    lines.extend(
        [
            "",
            "## Bucket Results",
            "",
            "| Bucket | Tasks | Wins | Losses | Ties | Avg Score Delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket, summary in _bucket_summaries(result.evaluation).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    bucket,
                    str(summary["tasks"]),
                    str(summary["wins"]),
                    str(summary["losses"]),
                    str(summary["ties"]),
                    _format_float(summary["avg_score_delta"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Failure Examples",
            "",
        ]
    )
    failures = [
        task_result
        for task_result in result.evaluation.results
        if task_result.outcome is PairOutcome.LOSS or task_result.candidate_run.error_json
    ]
    if not failures:
        lines.append("- No candidate failures observed.")
    else:
        for task_result in failures[:5]:
            lines.append(
                "- "
                + f"`{task_result.task_id}` bucket=`{task_result.bucket}` "
                + f"score_delta={_format_float(task_result.score_delta)}"
            )

    lines.extend(
        [
            "",
            "## Task Results",
            "",
            "| Task | Bucket | Outcome | Champion Score | Candidate Score | Score Delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for task_result in result.evaluation.results:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{task_result.task_id}`",
                    task_result.bucket,
                    task_result.outcome.value,
                    _format_float(task_result.champion_run.score),
                    _format_float(task_result.candidate_run.score),
                    _format_float(task_result.score_delta),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_full_eval_report(
    result: FullEvaluationResult,
    report_directory: str | Path,
) -> FullEvaluationResult:
    """Write a Full Eval Markdown report and return result with report_path."""
    target_directory = Path(report_directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    report_path = target_directory / f"{result.experiment_id}.md"
    report_path.write_text(render_full_eval_report(result), encoding="utf-8")
    return result.model_copy(update={"report_path": report_path})


def full_eval_result_to_experiment(result: FullEvaluationResult) -> EvalExperiment:
    """Convert a Full Eval result into a persisted experiment summary."""
    return EvalExperiment(
        id=result.experiment_id,
        champion_id=result.evaluation.champion_strategy_id,
        candidate_id=result.evaluation.candidate_strategy_id,
        task_set_id=result.task_set_id,
        quick_reject_passed=True,
        win_rate=result.evaluation.win_rate,
        p_value=result.significance.p_value,
        avg_score_delta=result.evaluation.avg_score_delta,
        cost_delta=result.cost_delta,
        latency_delta=result.latency_delta,
        safety_violation_count=result.safety_violation_count,
        critical_bucket_regression=result.critical_bucket_regression,
        verdict=result.verdict.value,
        report_path=result.report_path,
    )


def make_full_eval_experiment_id(
    *,
    champion_strategy_id: str,
    candidate_strategy_id: str,
) -> str:
    """Build a stable experiment id for Full Eval runs."""
    return f"full-eval-{champion_strategy_id}-vs-{candidate_strategy_id}"


def _validate_full_eval_config(
    *,
    rounds: int,
    min_win_rate: float,
    max_p_value: float,
    max_cost_delta: float,
    max_latency_delta: float,
    tie_threshold: float,
) -> None:
    """Reject invalid Full Eval settings."""
    if rounds <= 0:
        raise FullEvalError("rounds must be greater than 0")
    if not 0 <= min_win_rate <= 1:
        raise FullEvalError("min_win_rate must be between 0 and 1")
    if not 0 <= max_p_value <= 1:
        raise FullEvalError("max_p_value must be between 0 and 1")
    if max_cost_delta < 0:
        raise FullEvalError("max_cost_delta must be greater than or equal to 0")
    if max_latency_delta < 0:
        raise FullEvalError("max_latency_delta must be greater than or equal to 0")
    if tie_threshold < 0:
        raise FullEvalError("tie_threshold must be greater than or equal to 0")


def _full_eval_failed_checks(
    *,
    evaluation: PairedEvaluationResult,
    significance: SignificanceResult,
    cost_delta: float,
    latency_delta: float,
    safety_violation_count: int,
    critical_bucket_regression: bool,
    min_win_rate: float,
    min_score_delta: float,
    max_cost_delta: float,
    max_latency_delta: float,
    require_no_safety_violation: bool,
    require_no_critical_bucket_regression: bool,
) -> list[str]:
    """Return failed Full Eval gate names."""
    failed_checks: list[str] = []
    if evaluation.win_rate < min_win_rate:
        failed_checks.append("win_rate_below_threshold")
    if evaluation.avg_score_delta <= min_score_delta:
        failed_checks.append("avg_score_delta_below_threshold")
    if significance.insufficient_sample:
        failed_checks.append("insufficient_sample")
    elif significance.p_value is not None and significance.p_value > significance.alpha:
        failed_checks.append("p_value_above_threshold")
    if cost_delta > max_cost_delta:
        failed_checks.append("cost_delta_above_threshold")
    if latency_delta > max_latency_delta:
        failed_checks.append("latency_delta_above_threshold")
    if require_no_safety_violation and safety_violation_count > 0:
        failed_checks.append("safety_violation")
    if require_no_critical_bucket_regression and critical_bucket_regression:
        failed_checks.append("critical_bucket_regression")
    return failed_checks


def _bucket_summaries(
    evaluation: PairedEvaluationResult,
) -> dict[str, dict[str, int | float]]:
    """Aggregate paired results by task bucket."""
    grouped_results = defaultdict(list)
    for task_result in evaluation.results:
        grouped_results[task_result.bucket].append(task_result)

    summaries: dict[str, dict[str, int | float]] = {}
    for bucket in sorted(grouped_results):
        task_results = grouped_results[bucket]
        summaries[bucket] = {
            "tasks": len(task_results),
            "wins": sum(1 for item in task_results if item.outcome is PairOutcome.WIN),
            "losses": sum(1 for item in task_results if item.outcome is PairOutcome.LOSS),
            "ties": sum(1 for item in task_results if item.outcome is PairOutcome.TIE),
            "avg_score_delta": _average([item.score_delta for item in task_results]),
        }
    return summaries


def _has_critical_bucket_regression(
    evaluation: PairedEvaluationResult,
    *,
    tie_threshold: float,
) -> bool:
    """Return whether any critical task regressed beyond the tie threshold."""
    return any(
        result.bucket == "critical" and result.score_delta < -tie_threshold
        for result in evaluation.results
    )


def _candidate_safety_violation_count(evaluation: PairedEvaluationResult) -> int:
    """Count candidate runs with safety violations in their score breakdown."""
    return sum(
        1
        for result in evaluation.results
        if result.candidate_run.score_breakdown.get("safety_penalty", 0.0) > 0.0
        or result.candidate_run.score_breakdown.get("safety", 1.0) < 1.0
    )


def _relative_delta(baseline: float, candidate: float) -> float:
    """Return relative delta from baseline to candidate."""
    if baseline == 0:
        return 0.0 if candidate == 0 else 1.0
    return (candidate - baseline) / baseline


def _average(values: Sequence[float | int]) -> float:
    """Return the arithmetic mean for a non-empty sequence."""
    return sum(float(value) for value in values) / len(values)


def _format_float(value: float | int) -> str:
    """Format a numeric metric for reports."""
    return f"{float(value):.4f}"


def _format_percent(value: float | int) -> str:
    """Format a relative delta as a percentage."""
    return f"{float(value) * 100:.2f}%"


def _format_optional_float(value: float | None) -> str:
    """Format a nullable float for reports."""
    if value is None:
        return "n/a"
    return _format_float(value)
