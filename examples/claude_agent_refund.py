"""stategate + Claude Agent SDK.

Setup:
    pip install "stategate[claude]"
    # The Claude Agent SDK also needs its CLI runtime - see the SDK's install docs.
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/claude_agent_refund.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
)
from fake_shop import CONVERSATION, INSTRUCTIONS, check_code, find_order, refund

from stategate import Guard, Policy
from stategate.adapters.claude_agent_sdk import guard_hooks, guarded_tool, state_tool

POLICY = Policy.from_file(Path(__file__).parent / "refund_policy.yaml")


def build_options(guard: Guard) -> ClaudeAgentOptions:
    @guarded_tool(
        guard,
        "verify_identity",
        "Verify the customer's identity with their one-time code.",
        {"email": str, "code": str},
    )
    async def verify_identity(args):
        return check_code(args["email"], args["code"])  # a plain dict is fine

    @guarded_tool(guard, "lookup_order", "Look up an order by id.", {"order_id": str})
    async def lookup_order(args):
        return find_order(args["order_id"])

    @guarded_tool(guard, "issue_refund", "Refund money for an order.", {"order_id": str, "amount": float})
    async def issue_refund(args):
        return refund(args["order_id"], args["amount"])

    server = create_sdk_mcp_server("shop", tools=[verify_identity, lookup_order, issue_refund, state_tool(guard)])
    tool_names = [f"mcp__shop__{n}" for n in ("verify_identity", "lookup_order", "issue_refund", "get_trusted_state")]
    return ClaudeAgentOptions(
        system_prompt=INSTRUCTIONS,
        mcp_servers={"shop": server},
        tools=[],  # no built-in tools (Bash, files...) for this agent
        allowed_tools=tool_names,  # our tools run without a permission prompt; stategate guards them
        # The hooks are a second safety net: any other tool the agent might reach is
        # denied, because the policy says `unlisted_tools: deny`.
        hooks=guard_hooks(guard),
        max_turns=15,
    )


async def main() -> None:
    guard = Guard(POLICY)  # one guard per conversation
    async with ClaudeSDKClient(options=build_options(guard)) as client:
        for user_message in CONVERSATION:
            print(f"\nCUSTOMER: {user_message}")
            await client.query(user_message)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"AGENT: {block.text}")

    print("\n--- audit log ---")
    for event in guard.audit_log():
        print(json.dumps({k: event[k] for k in ("event", "tool", "args", "reasons")}))
    print("\n--- final trusted state ---")
    print(json.dumps(guard.state.snapshot(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
