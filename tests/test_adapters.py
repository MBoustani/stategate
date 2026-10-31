"""Adapter tests run against the real SDK packages (skipped if not installed).

They call the tools the same way the SDKs do, without contacting any model API.
"""

import asyncio
import json

import pytest

from stategate import Guard


def test_openai_agents_adapter(refund_policy):
    pytest.importorskip("agents")
    from agents import FunctionTool, RunContextWrapper
    from agents.tool_context import ToolContext

    from stategate.adapters.openai_agents import guarded_function_tool, state_tool

    guard = Guard(refund_policy)

    def verify_identity(email: str, code: str) -> dict:
        """Verify the customer's identity with the code they received.

        Args:
            email: Customer email.
            code: One-time code.
        """
        return {"verified": code == "123456"}

    @guarded_function_tool(guard)
    def lookup_order(ctx: RunContextWrapper, order_id: str) -> dict:
        """Look up an order."""
        assert ctx is not None  # the SDK still injects its context object
        return {"found": True, "order_id": order_id, "total": 10.0}

    verify_tool = guarded_function_tool(guard, verify_identity)
    status_tool = state_tool(guard)

    assert isinstance(verify_tool, FunctionTool)
    assert verify_tool.name == "verify_identity"
    assert set(verify_tool.params_json_schema["properties"]) == {"email", "code"}
    assert "identity" in verify_tool.description
    # context parameter is not part of the schema the model sees
    assert set(lookup_order.params_json_schema["properties"]) == {"order_id"}

    def invoke(tool, arguments):
        ctx = ToolContext(
            context=None, tool_name=tool.name, tool_call_id="call_1", tool_arguments=json.dumps(arguments)
        )
        return asyncio.run(tool.on_invoke_tool(ctx, json.dumps(arguments)))

    blocked = invoke(lookup_order, {"order_id": "A-1"})
    assert "BLOCKED" in str(blocked)

    invoke(verify_tool, {"email": "a@b.c", "code": "123456"})
    assert guard.state["identity_verified"] is True

    invoke(lookup_order, {"order_id": "A-1"})
    assert guard.state["order_id"] == "A-1"
    # the context object was not recorded as a tool argument
    assert guard.audit[-1].args == {"order_id": "A-1"}

    status = json.loads(invoke(status_tool, {}))
    assert status["tools"]["issue_refund"]["available_now"] is True


def test_claude_agent_sdk_guarded_tool(refund_policy):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

    from stategate.adapters.claude_agent_sdk import guarded_tool, state_tool

    guard = Guard(refund_policy)

    @guarded_tool(guard, "verify_identity", "Verify identity", {"email": str, "code": str})
    async def verify_identity(args):
        return {"verified": args["code"] == "123456"}  # plain dict

    @guarded_tool(guard, "lookup_order", "Look up an order", {"order_id": str})
    async def lookup_order(args):
        # classic MCP format with JSON text also works for effects
        data = {"found": True, "order_id": args["order_id"], "total": 42.0}
        return {"content": [{"type": "text", "text": json.dumps(data)}]}

    status = state_tool(guard)
    assert isinstance(verify_identity, SdkMcpTool)
    create_sdk_mcp_server("support", tools=[verify_identity, lookup_order, status])

    blocked = asyncio.run(lookup_order.handler({"order_id": "A-1"}))
    assert blocked["is_error"] is True
    assert "BLOCKED" in blocked["content"][0]["text"]

    ok = asyncio.run(verify_identity.handler({"email": "a@b.c", "code": "123456"}))
    assert json.loads(ok["content"][0]["text"]) == {"verified": True}
    assert guard.state["identity_verified"] is True

    asyncio.run(lookup_order.handler({"order_id": "A-1"}))
    assert guard.state["order_total"] == 42.0

    status_out = asyncio.run(status.handler({}))
    assert json.loads(status_out["content"][0]["text"])["state"]["order_id"]["value"] == "A-1"


def test_claude_agent_sdk_error_result_applies_no_effects(refund_policy):
    pytest.importorskip("claude_agent_sdk")
    from stategate.adapters.claude_agent_sdk import guarded_tool

    guard = Guard(refund_policy)

    @guarded_tool(guard, "verify_identity", "Verify identity", {"email": str, "code": str})
    async def verify_identity(args):
        return {"content": [{"type": "text", "text": "service down"}], "is_error": True}

    asyncio.run(verify_identity.handler({"email": "a", "code": "1"}))
    assert guard.call_count("verify_identity") == 0
    assert guard.audit[-1].event == "failed"


def test_claude_agent_sdk_hooks(refund_policy):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions

    from stategate.adapters.claude_agent_sdk import guard_hooks, policy_name

    assert policy_name("mcp__support__issue_refund") == "issue_refund"
    assert policy_name("Bash") == "Bash"

    guard = Guard(refund_policy)
    hooks = guard_hooks(guard)
    ClaudeAgentOptions(hooks=hooks)  # accepted by the SDK
    pre = hooks["PreToolUse"][0].hooks[0]
    post = hooks["PostToolUse"][0].hooks[0]

    def event(name, tool_input, **extra):
        return {"tool_name": name, "tool_input": tool_input, "tool_use_id": "tu_1", **extra}

    denied = asyncio.run(pre(event("mcp__support__lookup_order", {"order_id": "A-1"}), "tu_1", None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "verify_identity" in denied["hookSpecificOutput"]["permissionDecisionReason"]

    # an unlisted built-in tool is denied because the policy says unlisted_tools: deny
    denied_bash = asyncio.run(pre(event("Bash", {"command": "rm -rf /"}), "tu_2", None))
    assert denied_bash["hookSpecificOutput"]["permissionDecision"] == "deny"

    allowed = asyncio.run(pre(event("mcp__support__verify_identity", {"email": "a", "code": "1"}), "tu_3", None))
    assert allowed == {}
    response = [{"type": "text", "text": '{"verified": true}'}]
    asyncio.run(
        post(event("mcp__support__verify_identity", {"email": "a", "code": "1"}, tool_response=response), "tu_3", None)
    )
    assert guard.state["identity_verified"] is True
    assert guard.check("lookup_order").allowed
