from pathlib import Path

import pytest

from goagentx.core.strategy import (
    Genome,
    ModelGenome,
    PromptGenome,
    Strategy,
    StrategyStatus,
    ToolsGenome,
)
from goagentx.promotion.controller import PromotionController, PromotionControllerError
from goagentx.promotion.gate import PromotionDecision, PromotionGateMetrics, PromotionGateResult
from goagentx.registry.strategy_registry import StrategyRegistry


def test_promotion_controller_advances_candidate_to_shadow_and_writes_event(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))
    controller = PromotionController(registry)

    result = controller.promote(
        candidate.id,
        target_status=StrategyStatus.SHADOW,
        gate=_approved_gate(candidate.id),
        reason="full_eval_passed",
    )

    events = controller.list_events(strategy_id=candidate.id)

    assert result.strategy.status is StrategyStatus.SHADOW
    assert result.event.from_status is StrategyStatus.CANDIDATE
    assert result.event.to_status is StrategyStatus.SHADOW
    assert result.event.reason == "full_eval_passed"
    assert result.event.experiment_id == "eval-candidate-001"
    assert registry.get(candidate.id).status is StrategyStatus.SHADOW
    assert events == [result.event]


def test_promotion_controller_supports_full_shadow_canary_champion_path(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    old_champion = registry.create(
        _strategy("champion-old", StrategyStatus.CHAMPION, task_type="doc_qa")
    )
    candidate = registry.create(
        _strategy("candidate-001", StrategyStatus.CANDIDATE, task_type="doc_qa")
    )
    controller = PromotionController(registry)

    controller.promote(
        candidate.id,
        target_status=StrategyStatus.SHADOW,
        gate=_approved_gate(candidate.id),
    )
    controller.promote(
        candidate.id,
        target_status=StrategyStatus.CANARY,
        gate=_approved_gate(candidate.id),
    )
    controller.promote(
        candidate.id,
        target_status=StrategyStatus.CHAMPION,
        gate=_approved_gate(candidate.id),
    )

    events = controller.list_events(strategy_id=candidate.id)

    assert registry.get(candidate.id).status is StrategyStatus.CHAMPION
    assert registry.get(old_champion.id).status is StrategyStatus.RETIRED
    assert [event.from_status for event in events] == [
        StrategyStatus.CANDIDATE,
        StrategyStatus.SHADOW,
        StrategyStatus.CANARY,
    ]
    assert [event.to_status for event in events] == [
        StrategyStatus.SHADOW,
        StrategyStatus.CANARY,
        StrategyStatus.CHAMPION,
    ]


def test_promotion_controller_rejects_direct_champion_promotion(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))
    controller = PromotionController(registry)

    with pytest.raises(PromotionControllerError, match="candidate.*champion"):
        controller.promote(
            candidate.id,
            target_status=StrategyStatus.CHAMPION,
            gate=_approved_gate(candidate.id),
        )

    assert registry.get(candidate.id).status is StrategyStatus.CANDIDATE
    assert controller.list_events(strategy_id=candidate.id) == []


def test_promotion_controller_rejects_gate_failure(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))
    controller = PromotionController(registry)

    with pytest.raises(PromotionControllerError, match="promotion gate rejected"):
        controller.promote(
            candidate.id,
            target_status=StrategyStatus.SHADOW,
            gate=_rejected_gate(candidate.id),
        )

    assert registry.get(candidate.id).status is StrategyStatus.CANDIDATE
    assert controller.list_events(strategy_id=candidate.id) == []


def test_promotion_controller_rejects_gate_for_different_candidate(
    tmp_path: Path,
) -> None:
    registry = StrategyRegistry(tmp_path / "goagentx.db")
    candidate = registry.create(_strategy("candidate-001", StrategyStatus.CANDIDATE))
    controller = PromotionController(registry)

    with pytest.raises(PromotionControllerError, match="does not match"):
        controller.promote(
            candidate.id,
            target_status=StrategyStatus.SHADOW,
            gate=_approved_gate("candidate-other"),
        )

    assert registry.get(candidate.id).status is StrategyStatus.CANDIDATE
    assert controller.list_events(strategy_id=candidate.id) == []


def _approved_gate(candidate_id: str) -> PromotionGateResult:
    return PromotionGateResult(
        decision=PromotionDecision.APPROVE,
        approved=True,
        failed_checks=[],
        metrics=_metrics(candidate_id),
    )


def _rejected_gate(candidate_id: str) -> PromotionGateResult:
    return PromotionGateResult(
        decision=PromotionDecision.REJECT,
        approved=False,
        failed_checks=["win_rate_below_threshold"],
        metrics=_metrics(candidate_id, win_rate=0.1),
    )


def _metrics(candidate_id: str, **overrides: object) -> PromotionGateMetrics:
    data = {
        "experiment_id": f"eval-{candidate_id}",
        "champion_id": "champion-old",
        "candidate_id": candidate_id,
        "win_rate": 0.8,
        "p_value": 0.01,
        "avg_score_delta": 0.1,
        "cost_delta": 0.05,
        "latency_delta": 0.05,
        "safety_violation_count": 0,
        "critical_bucket_regression": False,
    }
    data.update(overrides)
    return PromotionGateMetrics.model_validate(data)


def _strategy(
    strategy_id: str,
    status: StrategyStatus,
    *,
    task_type: str | None = "doc_qa",
) -> Strategy:
    return Strategy(
        id=strategy_id,
        version=1,
        name=strategy_id,
        task_type=task_type,
        status=status,
        genome=_sample_genome(),
    )


def _sample_genome() -> Genome:
    return Genome(
        model=ModelGenome(
            provider="openai_compatible",
            name="gpt-4.1",
            temperature=0.4,
            top_p=0.9,
        ),
        prompt_genome=PromptGenome(
            role="senior_code_reviewer",
            reasoning_style="evidence_first",
            risk_policy="strict",
            output_format="findings_first",
        ),
        tools=ToolsGenome(enabled=["repo_search", "shell_readonly", "browser"]),
    )
