"""stategate - deterministic tool guardrails for AI agents.

Prompts are suggestions; stategate is enforcement. Each tool declares what must be
true (in a trusted state that only code can change) before it may run. If the agent
calls a tool too early, the call is blocked and the agent is told exactly which
tools to call first.

Quick start::

    from stategate import Guard, Policy

    guard = Guard(Policy.from_file("policy.yaml"))

    @guard.tool
    def issue_refund(order_id: str, amount: float) -> dict:
        ...
"""

import logging as _logging

from ._version import __version__
from .checker import Issue, check_policy
from .conditions import Condition, ConditionSyntaxError
from .guard import AuditEvent, Decision, Guard, StateChange, ToolBlocked, Unmet
from .planner import Plan, suggest_plan, tool_edges
from .policy import Effect, Policy, PolicyError, StateVar, ToolSpec
from .render import to_markdown, to_mermaid
from .state import StateTypeError, TrustedState

# Libraries should stay quiet unless the application configures logging.
# To see stategate's decisions: logging.getLogger("stategate").setLevel(logging.INFO)
_logging.getLogger("stategate").addHandler(_logging.NullHandler())

__all__ = [
    "AuditEvent",
    "Condition",
    "ConditionSyntaxError",
    "Decision",
    "Effect",
    "Guard",
    "Issue",
    "Plan",
    "Policy",
    "PolicyError",
    "StateChange",
    "StateTypeError",
    "StateVar",
    "ToolBlocked",
    "ToolSpec",
    "TrustedState",
    "Unmet",
    "__version__",
    "check_policy",
    "suggest_plan",
    "to_markdown",
    "to_mermaid",
    "tool_edges",
]
