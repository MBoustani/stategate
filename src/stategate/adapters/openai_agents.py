"""OpenAI Agents SDK integration.

Use :func:`guarded_function_tool` wherever you would use ``agents.function_tool``::

    from agents import Agent
    from stategate import Guard, Policy
    from stategate.adapters.openai_agents import guarded_function_tool, state_tool

    guard = Guard(Policy.from_file("policy.yaml"))   # one guard per conversation

    def issue_refund(order_id: str, amount: float) -> dict:
        \"\"\"Refund an order.\"\"\"
        ...

    agent = Agent(
        name="support",
        tools=[guarded_function_tool(guard, issue_refund), state_tool(guard)],
    )

How it works: the SDK builds the tool's JSON schema from your function's
signature and docstring exactly as usual (the guard keeps both). When the model
calls the tool, the guard checks the policy first. If the call is blocked your
function is not run, and the model receives the explanation as the tool result,
so it can recover by calling the right tools first.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

try:
    from agents import FunctionTool, RunContextWrapper, function_tool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The OpenAI Agents SDK is not installed. Install it with: pip install openai-agents"
    ) from exc

from ..guard import Guard


def _context_params(func: Callable[..., Any]) -> tuple[str, ...]:
    """Names of parameters the SDK fills in itself (RunContextWrapper / ToolContext)."""
    params = list(inspect.signature(func).parameters.values())
    if not params:
        return ()
    first = params[0]
    try:
        annotation = get_type_hints(func).get(first.name, first.annotation)
    except Exception:
        annotation = first.annotation
    origin = getattr(annotation, "__origin__", annotation)
    if inspect.isclass(origin) and issubclass(origin, RunContextWrapper):
        return (first.name,)
    return ()


def guarded_function_tool(
    guard: Guard,
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    **function_tool_kwargs: Any,
) -> Any:
    """Like ``agents.function_tool``, but every call goes through ``guard``.

    ``name`` is both the tool name the model sees and the name used in the policy
    (defaults to the function name). Extra keyword arguments go to ``function_tool``.
    Can be used as ``guarded_function_tool(guard, fn)`` or as a decorator
    ``@guarded_function_tool(guard)``.
    """

    def build(fn: Callable[..., Any]) -> FunctionTool:
        tool_name = name or fn.__name__
        wrapped = guard.wrap(fn, name=tool_name, skip_params=_context_params(fn))
        return function_tool(wrapped, name_override=tool_name, **function_tool_kwargs)

    if func is None:
        return build
    return build(func)


def state_tool(guard: Guard, *, name: str = "get_trusted_state") -> FunctionTool:
    """A read-only tool that shows the agent the trusted state and which tools are available.

    Add ``get_trusted_state: {}`` to your policy's ``tools`` (or set
    ``unlisted_tools: allow``) so the agent is allowed to call it.
    """

    def get_trusted_state() -> str:
        """Read the trusted session state and see which tools are available right now and what each one is waiting for. Read-only."""
        return guard.status_text()

    return guarded_function_tool(guard, get_trusted_state, name=name)
