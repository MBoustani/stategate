import pytest

from stategate.conditions import MISSING, Condition, ConditionSyntaxError, Ref, resolve


@pytest.mark.parametrize(
    "text, op, right",
    [
        ("state.verified == true", "==", True),
        ("state.verified==false", "==", False),
        ("state.count >= 3", ">=", 3),
        ("state.total < 10.5", "<", 10.5),
        ('state.status == "found"', "==", "found"),
        ("state.status == found", "==", "found"),
        ("state.status == 'found'", "==", "found"),
        ("state.status != null", "!=", None),
        ('state.status in ["a", "b"]', "in", ["a", "b"]),
        ("state.status not   in [1, 2]", "not in", [1, 2]),
        ("state.order_id is set", "is set", None),
        ("state.order_id is  not set", "is not set", None),
    ],
)
def test_parse(text, op, right):
    cond = Condition.parse(text)
    assert cond.op == op
    assert cond.right == right
    assert cond.left.namespace == "state"


def test_parse_reference_on_the_right():
    cond = Condition.parse("args.amount <= $state.order_total")
    assert cond.left == Ref("args", ("amount",))
    assert cond.right == Ref("state", ("order_total",))
    assert cond.namespaces() == {"args", "state"}


@pytest.mark.parametrize(
    "text",
    [
        "verified == true",  # missing namespace
        "state.verified",  # no operator
        "state.status in 'abc'",  # 'in' needs a list
        "state. == 1",
        "foo.bar == 1",
        "state.x == $nope.y",
    ],
)
def test_parse_errors(text):
    with pytest.raises(ConditionSyntaxError):
        Condition.parse(text)


def test_evaluate_basic():
    ctx = {"state": {"verified": True, "total": 100.0, "status": "found"}, "args": {"amount": 50}}
    assert Condition.parse("state.verified == true").evaluate(ctx)
    assert not Condition.parse("state.verified == false").evaluate(ctx)
    assert Condition.parse("args.amount <= $state.total").evaluate(ctx)
    assert not Condition.parse("args.amount > $state.total").evaluate(ctx)
    assert Condition.parse('state.status in ["found", "shipped"]').evaluate(ctx)
    assert Condition.parse('state.status not in ["refunded"]').evaluate(ctx)


def test_missing_values_fail_closed():
    ctx = {"state": {}, "args": {}}
    assert not Condition.parse("state.total > 0").evaluate(ctx)
    assert not Condition.parse("args.amount <= $state.total").evaluate(ctx)
    assert not Condition.parse("state.x is set").evaluate(ctx)
    assert Condition.parse("state.x is not set").evaluate(ctx)
    assert not Condition.parse("state.x in $state.y").evaluate(ctx)


def test_type_mismatch_is_false_not_an_exception():
    ctx = {"state": {"total": "abc"}, "args": {"amount": 5}}
    assert not Condition.parse("args.amount <= $state.total").evaluate(ctx)
    # bools are not numbers here
    assert not Condition.parse("state.flag < 5").evaluate({"state": {"flag": True}})


def test_nested_paths():
    ctx = {"result": {"order": {"id": "A1"}}}
    assert resolve(Ref.parse("result.order.id"), ctx) == "A1"
    assert resolve(Ref.parse("result.order.nope"), ctx) is MISSING
    assert resolve(Ref.parse("result"), {"result": 42}) == 42


def test_explain_shows_current_values():
    cond = Condition.parse("args.amount <= $state.total")
    text = cond.explain({"state": {"total": 10}, "args": {"amount": 99}})
    assert "args.amount is 99" in text and "state.total is 10" in text
