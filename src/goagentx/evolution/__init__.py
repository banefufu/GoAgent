"""Evolution workflows for GoAgentX."""

from goagentx.evolution.scheduler import (
    DegradationDetector,
    DegradationResult,
    detect_score_degradation,
)

__all__ = [
    "DegradationDetector",
    "DegradationResult",
    "detect_score_degradation",
]
