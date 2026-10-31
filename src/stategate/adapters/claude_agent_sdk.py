"""Claude Agent SDK integration.

Two ways to plug stategate in - pick ONE per tool:

1. **Your own tools** - use :func:`guarded_tool` instead of ``claude_agent_sdk.tool``::

       from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
       from stategate import Guard, Policy
       from stategate.adapters.claude_agent_sdk import guarded_tool, state_tool

       guard = Guard(Policy.from_file("policy.yaml"))   # one guard per conversation

       @guarded_tool(guard, "issue_refund", "Refund an order", {"order_id": str, "amount": float})
       async def issue_refund(args):
           return {"refunded": True}          # a plain dict is fine, see below

       server = create_sdk_mcp_server("support", tools=[issue_refund, state_tool(guard)])

   Your handler may return a plain dict. stategate uses that dict for the
   policy's ``result.*`` effects and sends it to the model as JSON text. (You can
   still return the usual ``{"content": [...]}`` MCP format if you prefer.)

2. **Built-in tools** (Bash, Write, Edit, WebFetch ...) - use :func:`guard_hooks`::

       options = ClaudeAgentOptions(hooks=guard_hooks(guard), ...)

   This registers ``PreToolUse`` (check / deny) and ``PostToolUse`` (apply
   effects) hooks. Tools already wrapped with :func:`guarded_tool` are skipped by
   the hooks so they are never checked twice. Tool names like
   ``mcp__support__issue_refund`` are matched to ``issue_refund`` in the policy.
   Remember to set ``unlisted_tools: allow`` in the policy if you only want to
   guard a few built-in tools and leave the rest alone.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

try:
    from claude_agent_sdk import HookMatcher, SdkMcpTool, tool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Claude Agent SDK is not installed. Install it with: pip install 'stategate[claude]'"
    ) from exc

from ..guard import Guard

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["is_error"] = True
    return result


def _structured(value: Any) -> Any:
    """Get structured data out of an MCP-style result so ``result.<field>`` works."""
    content = None
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        content = value["content"]
    elif isinstance(value, list):
        content = value
    if content is None:
        return value
    texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    joined = "".join(texts).strip()
    try:
        return json.loads(joined)
    except ValueError:
        return joined if texts else value


def guarded_tool(
    guard: Guard,
    name: str,
    description: str,
    input_schema: type | dict[str, Any],
    *,
    annotations: Any = None,
) -> Callable[[Handler], SdkMcpTool[Any]]:
    """Like ``claude_agent_sdk.tool``, but every call goes through ``guard``."""

    def decorator(handler: Handler) -> SdkMcpTool[Any]:
        guard.managed_tools.add(name)  # tell guard_hooks not to check this tool a second time

        async def guarded_handler(args: dict[str, Any]) -> dict[str, Any]:
            decision = guard.begin(name, args)
            if not decision.allowed:
                return _text_result(decision.message(), is_error=True)
            try:
                raw = await handler(args)
            except BaseException as exc:
                guard.abort(name, args, exc, call_id=decision.call_id)
                raise
            is_mcp = isinstance(raw, dict) and "content" in raw
            if is_mcp and raw.get("is_error"):
                # The tool reported an error itself: don't apply effects.
                guard.abort(name, args, "tool returned is_error", call_id=decision.call_id)
                return raw
            guard.finish(name, args, _structured(raw) if is_mcp else raw, call_id=decision.call_id)
            if is_mcp:
                return raw
            text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            return _text_result(text)

        kwargs = {"annotations": annotations} if annotations is not None else {}
        return tool(name, description, input_schema, **kwargs)(guarded_handler)

    return decorator


def state_tool(guard: Guard, *, name: str = "get_trusted_state") -> SdkMcpTool[Any]:
    """A read-only tool that shows Claude the trusted state and which tools are available."""

    @guarded_tool(
        guard,
        name,
        "Read the trusted session state and see which tools are available right now and what each one is waiting for. Read-only.",
        {},
    )
    async def get_trusted_state(args: dict[str, Any]) -> str:
        return guard.status_text()

    return get_trusted_state


def policy_name(tool_name: str) -> str:
    """``mcp__server__issue_refund`` -> ``issue_refund``; other names unchanged."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return tool_name


def guard_hooks(guard: Guard, *, map_name: Callable[[str], str] = policy_name) -> dict[str, list[HookMatcher]]:
    """Hooks for ``ClaudeAgentOptions(hooks=...)`` that enforce ``guard`` on any tool."""

    def should_skip(name: str) -> bool:
        return name in guard.managed_tools  # handled by guarded_tool already

    async def pre_tool_use(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        name = map_name(input_data.get("tool_name", ""))
        if should_skip(name):
            return {}
        args = input_data.get("tool_input") or {}
        call_id = tool_use_id or input_data.get("tool_use_id")
        decision = guard.begin(name, args, call_id=call_id)
        if decision.allowed:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision.message(),
            }
        }

    async def post_tool_use(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        name = map_name(input_data.get("tool_name", ""))
        if should_skip(name):
            return {}
        args = input_data.get("tool_input") or {}
        call_id = tool_use_id or input_data.get("tool_use_id")
        guard.finish(name, args, _structured(input_data.get("tool_response")), call_id=call_id)
        return {}

    async def post_tool_use_failure(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        name = map_name(input_data.get("tool_name", ""))
        if should_skip(name):
            return {}
        args = input_data.get("tool_input") or {}
        call_id = tool_use_id or input_data.get("tool_use_id")
        guard.abort(name, args, str(input_data.get("error", "tool failed")), call_id=call_id)
        return {}

    return {
        "PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use])],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[post_tool_use])],
        "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[post_tool_use_failure])],
    }
