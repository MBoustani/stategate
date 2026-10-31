import json

import pytest

from stategate import Guard, Policy, PolicyError, check_policy


def errors(policy):
    return [i for i in check_policy(policy) if i.severity == "error"]


def test_example_policy_is_clean(refund_policy):
    assert errors(refund_policy) == []
    infos = [i for i in check_policy(refund_policy) if i.severity == "info"]
    assert any("manager_approved" in i.where for i in infos)  # app-controlled state is reported


def test_state_shorthand_and_empty_tools():
    policy = Policy.from_yaml("state:\n  ok: bool\ntools:\n  a:\n")
    assert policy.state["ok"].type == "bool"
    assert policy.tools["a"].requires == []


def test_effect_needs_exactly_one_source():
    with pytest.raises(PolicyError, match="either 'value' or 'from'"):
        Policy.from_yaml("state: {x: int}\ntools:\n  t:\n    effects: [{set: x, value: 1, from: result.y}]\n")
    with pytest.raises(PolicyError, match="needs 'value' or 'from'"):
        Policy.from_yaml("state: {x: int}\ntools:\n  t:\n    effects: [{set: x}]\n")


def test_effect_value_null_means_unset():
    policy = Policy.from_yaml("state: {x: int}\ntools:\n  t:\n    effects: [{set: x, value: null}]\n")
    assert policy.tools["t"].effects[0].is_literal


def test_effect_from_must_be_result_or_args():
    with pytest.raises(PolicyError, match="'from' must start with"):
        Policy.from_yaml("state: {x: int, y: int}\ntools:\n  t:\n    effects: [{set: x, from: state.y}]\n")


def test_requires_cannot_use_result():
    with pytest.raises(PolicyError, match="has not run yet"):
        Policy.from_yaml("tools:\n  t:\n    requires: [result.ok == true]\n")


def test_reserved_state_names():
    with pytest.raises(PolicyError, match="reserved"):
        Policy.from_yaml("state: {args: int}\n")


def test_unknown_fields_are_rejected():
    with pytest.raises(PolicyError):
        Policy.from_yaml("tools:\n  t:\n    requirez: []\n")


def test_checker_finds_reference_and_type_errors():
    policy = Policy.from_yaml(
        """
state:
  verified: bool
  status: {type: str, enum: [a, b]}
tools:
  t1:
    after: [nope]
    requires:
      - state.verifed == true
      - state.status == c
      - state.verified == "yes"
    effects:
      - {set: missing, value: 1}
      - {set: verified, value: 3}
"""
    )
    messages = [str(i) for i in errors(policy)]
    joined = "\n".join(messages)
    assert "unknown tool 'nope'" in joined
    assert "unknown state variable 'verifed'" in joined
    assert "can never match" in joined
    assert "sets unknown state variable 'missing'" in joined
    assert "value 3 is not bool" in joined


def test_checker_detects_circular_dependency():
    policy = Policy.from_yaml(
        """
state:
  a_done: {type: bool, default: false}
  b_done: {type: bool, default: false}
tools:
  a:
    requires: [state.b_done == true]
    effects: [{set: a_done, value: true}]
  b:
    requires: [state.a_done == true]
    effects: [{set: b_done, value: true}]
"""
    )
    errs = errors(policy)
    assert {e.where for e in errs} == {"tools.a", "tools.b"}
    assert all("can never run" in e.message for e in errs)
    with pytest.raises(PolicyError, match="can never run"):
        Guard(policy)


def test_unsatisfiable_requirement_warns():
    policy = Policy.from_yaml(
        """
state:
  status: {type: str, default: new}
tools:
  setter:
    effects: [{set: status, value: started}]
  needs_done:
    requires: [state.status == done]
"""
    )
    warnings = [i for i in check_policy(policy) if i.severity == "warning"]
    assert any("no tool can make" in w.message for w in warnings)


def test_json_schema_is_valid_json():
    schema = Policy.json_schema()
    text = json.dumps(schema)
    assert "unlisted_tools" in text and "effects" in text
    assert schema["title"] == "stategate policy"


def test_roundtrip_yaml(refund_policy):
    again = Policy.from_yaml(refund_policy.to_yaml())
    assert again.model_dump() == refund_policy.model_dump()


def test_from_json_file(tmp_path, refund_policy):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(refund_policy.model_dump(by_alias=True)))
    assert Policy.from_file(path).tools.keys() == refund_policy.tools.keys()


def test_yaml_boolean_trap_gets_a_hint():
    policy = Policy.from_yaml(
        'state:\n  mode: {type: str, default: off}\ntools:\n  t:\n    requires: [state.mode == "x"]\n'
    )
    msgs = [i.message for i in errors(policy)]
    assert any("unquoted yes/no/on/off" in m for m in msgs)


def test_roundtrip_keeps_unset_effects():
    policy = Policy.from_yaml("state: {x: int}\ntools:\n  t:\n    effects: [{set: x, value: null}]\n")
    again = Policy.from_yaml(policy.to_yaml())
    assert again.tools["t"].effects[0].is_literal
    assert again.tools["t"].effects[0].value is None
