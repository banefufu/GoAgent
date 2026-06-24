"""Promotion and rollback controls for GoAgentX."""

from goagentx.promotion.gate import (
    PromotionDecision,
    PromotionGateMetrics,
    PromotionGateResult,
    evaluate_promotion_gate,
    promotion_metrics_from_full_eval,
)

__all__ = [
    "PromotionDecision",
    "PromotionGateMetrics",
    "PromotionGateResult",
    "evaluate_promotion_gate",
    "promotion_metrics_from_full_eval",
]
