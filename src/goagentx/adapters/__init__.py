"""Agent execution adapters for GoAgentX."""

from goagentx.adapters.agent_runner import FakeAgentRunner, StaticQualityRunner
from goagentx.core.run import AgentRunner, AgentRunResult

__all__ = [
    "AgentRunner",
    "AgentRunResult",
    "FakeAgentRunner",
    "StaticQualityRunner",
]
