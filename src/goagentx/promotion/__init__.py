"""Promotion and rollback controls for GoAgentX."""

from goagentx.promotion.controller import (
    PromotionController,
    PromotionControllerError,
    PromotionEvent,
    PromotionResult,
)
from goagentx.promotion.gate import (
    PromotionDecision,
    PromotionGateMetrics,
    PromotionGateResult,
    evaluate_promotion_gate,
    promotion_metrics_from_full_eval,
)

__all__ = [
    "PromotionController",
    "PromotionControllerError",
    "PromotionDecision",
    "PromotionEvent",
    "PromotionGateMetrics",
    "PromotionGateResult",
    "PromotionResult",
    "evaluate_promotion_gate",
    "promotion_metrics_from_full_eval",
]
