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
from goagentx.promotion.rollback import (
    RollbackController,
    RollbackControllerError,
    RollbackResult,
)

__all__ = [
    "PromotionController",
    "PromotionControllerError",
    "PromotionDecision",
    "PromotionEvent",
    "PromotionGateMetrics",
    "PromotionGateResult",
    "PromotionResult",
    "RollbackController",
    "RollbackControllerError",
    "RollbackResult",
    "evaluate_promotion_gate",
    "promotion_metrics_from_full_eval",
]
