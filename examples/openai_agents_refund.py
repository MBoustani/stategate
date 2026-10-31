"""stategate + OpenAI Agents SDK.

Setup:
    pip install "stategate[openai]"
    export OPENAI_API_KEY=sk-...
    python examples/openai_agents_refund.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agents import Agent, Runner
from fake_shop import CONVERSATION, INSTRUCTIONS, check_code, find_order, refund

from stategate import Guard, Policy
from stategate.adapters.openai_agents import guarded_function_tool, state_tool

POLICY = Policy.from_file(Path(__file__).parent / "refund_policy.yaml")


def build_agent(guard: Guard) -> Agent:
    # Plain functions with type hints and docstrings - the SDK turns them into tools as usual.
    def verify_identity(email: str, code: str) -> dict:
        """Verify the customer's identity with the one-time code they received by email.

        Args:
            email: The customer's email address.
            code: The 6-digit one-time code.
        """
        return check_code(email, code)

    def lookup_order(order_id: str) -> dict:
        """Look up an order by its id.

        Args:
            order_id: The order id, e.g. A-1001.
        """
        return find_order(order_id)

    def issue_refund(order_id: str, amount: float) -> dict:
        """Refund money for an order.

        Args:
            order_id: The order to refund.
            amount: Amount to refund.
        """
        return refund(order_id, amount)

    return Agent(
        name="Support agent",
        instructions=INSTRUCTIONS,
        tools=[
            guarded_function_tool(guard, verify_identity),
            guarded_function_tool(guard, lookup_order),
            guarded_function_tool(guard, issue_refund),
            state_tool(guard),  # lets the agent see the state and what each tool is waiting for
        ],
    )


async def main() -> None:
    guard = Guard(POLICY)  # one guard per conversation
    agent = build_agent(guard)

    history: list = []
    for user_message in CONVERSATION:
        print(f"\nCUSTOMER: {user_message}")
        result = await Runner.run(agent, history + [{"role": "user", "content": user_message}])
        print(f"AGENT: {result.final_output}")
        history = result.to_input_list()

    print("\n--- audit log ---")
    for event in guard.audit_log():
        print(json.dumps({k: event[k] for k in ("event", "tool", "args", "reasons")}))
    print("\n--- final trusted state ---")
    print(json.dumps(guard.state.snapshot(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
