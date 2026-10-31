import asyncio
import threading
import time

import pytest

from stategate import Guard, Policy, PolicyError, StateTypeError, ToolBlocked


def make_tools(guard):
    @guard.tool
    def verify_identity(email: str, code: str) -> dict:
        return {"verified": code == "123456"}

    @guard.tool
    def lookup_order(order_id: str) -> dict:
        if order_id == "A-1":
            return {"found": True, "order_id": "A-1", "total": 120.0}
        return {"found": False}

    @guard.tool
    def issue_refund(order_id: str, amount: float) -> dict:
        return {"refunded": amount}

    @guard.tool
    def send_confirmation_email() -> str:
        return "sent"

    @guard.tool
    def close_account() -> str:
        return "closed"

    return verify_identity, lookup_order, issue_refund, send_confirmation_email, close_account


def test_happy_path_and_effects(refund_policy):
    guard = Guard(refund_policy)
    verify, lookup, refund, confirm, _ = make_tools(guard)

    assert verify(email="a@b.c", code="123456") == {"verified": True}
    assert guard.state["identity_verified"] is True
    assert guard.state["customer_email"] == "a@b.c"

    assert lookup("A-1")["found"] is True
    assert guard.state["order_id"] == "A-1"
    assert guard.state["order_total"] == 120.0
    assert guard.state["order_status"] == "found"

    assert refund("A-1", 100.0) == {"refunded": 100.0}
    assert guard.state["order_status"] == "refunded"
    assert confirm() == "sent"


def test_blocked_call_does_not_run_the_function(refund_policy):
    guard = Guard(refund_policy)
    ran = []

    @guard.tool
    def issue_refund(order_id: str, amount: float) -> dict:
        ran.append(1)
        return {}

    out = issue_refund("A-1", 10)
    assert ran == []
    assert out.startswith("BLOCKED")
    assert "Suggested order: verify_identity -> lookup_order -> issue_refund" in out
    assert guard.audit[-1].event == "blocked"


def test_when_condition_prevents_state_update(refund_policy):
    guard = Guard(refund_policy)
    verify, lookup, *_ = make_tools(guard)
    verify(email="a@b.c", code="000000")  # wrong code: tool succeeds but says verified=False
    assert guard.state["identity_verified"] is False
    assert guard.state["customer_email"] is None
    assert lookup("A-1").startswith("BLOCKED")


def test_argument_rules(refund_policy):
    guard = Guard(refund_policy)
    verify, lookup, refund, *_ = make_tools(guard)
    verify(email="a@b.c", code="123456")
    lookup("A-1")

    wrong_order = refund("B-2", 10.0)
    assert "args.order_id == $state.order_id" in wrong_order
    assert '"B-2"' in wrong_order and '"A-1"' in wrong_order

    too_much = refund("A-1", 500.0)
    assert "args.amount <= $state.order_total" in too_much

    negative = refund("A-1", -5.0)
    assert "args.amount > 0" in negative

    assert refund("A-1", 120.0) == {"refunded": 120.0}


def test_max_calls(refund_policy):
    guard = Guard(
        refund_policy,
        initial_state={"identity_verified": True, "order_status": "found", "order_id": "A-1", "order_total": 50.0},
    )

    @guard.tool
    def issue_refund(order_id: str, amount: float) -> str:
        return "ok"

    assert issue_refund("A-1", 10.0) == "ok"
    # status became "refunded", and the max_calls limit is reached
    guard.set_state("order_status", "found")
    second = issue_refund("A-1", 10.0)
    assert "limit is reached" in second
    assert guard.call_count("issue_refund") == 1


def test_parallel_calls_cannot_exceed_max_calls():
    policy = Policy.from_yaml("tools:\n  pay:\n    max_calls: 1\n")
    guard = Guard(policy)
    results = []

    @guard.tool
    def pay() -> str:
        time.sleep(0.05)
        return "paid"

    threads = [threading.Thread(target=lambda: results.append(pay())) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("paid") == 1
    assert sum(r.startswith("BLOCKED") for r in results) == 4


def test_after_dependency(refund_policy):
    guard = Guard(refund_policy, initial_state={"order_status": "refunded"})
    *_, confirm, _ = make_tools(guard)
    out = confirm()
    assert "`issue_refund` must complete successfully" in out


def test_app_controlled_state(refund_policy):
    guard = Guard(refund_policy, initial_state={"identity_verified": True})
    *_, close = make_tools(guard)
    out = close()
    assert "controlled by the application" in out
    guard.set_state("manager_approved", True)  # e.g. after a human clicked "approve"
    assert close() == "closed"
    assert guard.audit[-2].event == "state_set"


def test_exception_means_no_effects_and_no_count(refund_policy):
    guard = Guard(refund_policy)

    @guard.tool
    def verify_identity(email: str, code: str) -> dict:
        raise RuntimeError("email service down")

    with pytest.raises(RuntimeError):
        verify_identity("a@b.c", "123456")
    assert guard.state["identity_verified"] is False
    assert guard.call_count("verify_identity") == 0
    assert guard.audit[-1].event == "failed"


def test_async_tools(refund_policy):
    guard = Guard(refund_policy)

    @guard.tool
    async def verify_identity(email: str, code: str) -> dict:
        await asyncio.sleep(0)
        return {"verified": True}

    @guard.tool
    async def lookup_order(order_id: str) -> dict:
        return {"found": True, "order_id": order_id, "total": 5.0}

    async def run():
        assert (await lookup_order("X")).startswith("BLOCKED")
        await verify_identity("a@b.c", "1")
        return await lookup_order("X")

    assert asyncio.run(run())["found"] is True
    assert guard.state["order_id"] == "X"


def test_unlisted_tools():
    deny = Guard(Policy.from_yaml("tools: {}\n"))
    with pytest.raises(PolicyError, match="not listed"):
        deny.wrap(lambda: 1, name="rogue")
    assert not deny.check("rogue").allowed

    allow = Guard(Policy.from_yaml("unlisted_tools: allow\n"))
    assert allow.wrap(lambda: 1, name="anything")() == 1


def test_state_is_read_only_and_typed(refund_policy):
    guard = Guard(refund_policy)
    with pytest.raises(TypeError):
        guard.state["identity_verified"] = True  # Mapping has no __setitem__
    with pytest.raises(StateTypeError):
        guard.set_state("identity_verified", "yes")
    with pytest.raises(StateTypeError):
        guard.set_state("order_status", "teleported")  # not in enum
    with pytest.raises(StateTypeError):
        guard.set_state("not_declared", 1)
    with pytest.raises(StateTypeError):
        Guard(refund_policy, initial_state={"order_total": "lots"})


def test_bad_tool_result_fails_closed(refund_policy, caplog):
    guard = Guard(refund_policy, initial_state={"identity_verified": True})

    @guard.tool
    def lookup_order(order_id: str) -> dict:
        return {"found": True, "order_id": "A-1", "total": "not a number"}

    lookup_order("A-1")
    assert guard.state["order_total"] is None  # wrong type was not stored
    assert "order_total" in caplog.text


def test_on_block_raise(refund_policy):
    guard = Guard(refund_policy, on_block="raise")

    @guard.tool
    def issue_refund(order_id: str, amount: float) -> str:
        return "ok"

    with pytest.raises(ToolBlocked) as info:
        issue_refund("A", 1.0)
    assert info.value.decision.tool == "issue_refund"
    assert not info.value.decision.allowed


def test_results_as_json_strings_and_objects(refund_policy):
    from dataclasses import dataclass

    guard = Guard(refund_policy)

    @guard.tool
    def verify_identity(email: str, code: str) -> str:
        return '{"verified": true}'

    verify_identity("a@b.c", "1")
    assert guard.state["identity_verified"] is True

    @dataclass
    class Order:
        found: bool
        order_id: str
        total: float

    @guard.tool
    def lookup_order(order_id: str) -> Order:
        return Order(True, order_id, 9.5)

    lookup_order("Z-9")
    assert guard.state["order_total"] == 9.5


def test_status_and_audit_are_json_friendly(refund_policy):
    import json

    guard = Guard(refund_policy)
    verify, *_ = make_tools(guard)
    verify("a@b.c", "123456")
    status = guard.status()
    assert status["tools"]["lookup_order"]["available_now"] is True
    assert status["tools"]["issue_refund"]["available_now"] is False
    assert status["state"]["identity_verified"]["value"] is True
    json.dumps(guard.audit_log())
    json.loads(guard.status_text())


def test_check_has_no_side_effects(refund_policy):
    guard = Guard(refund_policy)
    guard.check("issue_refund", {"order_id": "A", "amount": 1})
    assert guard.audit == []
