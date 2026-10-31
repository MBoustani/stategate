import json

from stategate import Policy, suggest_plan, to_markdown, to_mermaid, tool_edges
from stategate.cli import main


def test_plan_for_refund(refund_policy):
    plan = suggest_plan(refund_policy, "issue_refund", refund_policy.initial_state(), {})
    assert plan.steps == ["verify_identity", "lookup_order"]
    assert plan.unfixable == []


def test_plan_skips_what_is_already_done(refund_policy):
    plan = suggest_plan(refund_policy, "issue_refund", {"identity_verified": True, "order_status": "none"}, {})
    assert plan.steps == ["lookup_order"]


def test_plan_reports_app_controlled(refund_policy):
    plan = suggest_plan(refund_policy, "close_account", refund_policy.initial_state(), {})
    assert plan.steps == ["verify_identity"]
    assert any("controlled by the application" in r for r in plan.unfixable)


def test_plan_respects_exhausted_tools(refund_policy):
    state = {"identity_verified": True, "order_status": "found"}
    plan = suggest_plan(refund_policy, "send_confirmation_email", state, {"issue_refund": 1})
    assert "issue_refund" not in plan.steps
    assert plan.unfixable


def test_plan_prefers_definite_providers():
    policy = Policy.from_yaml(
        """
state:
  mode: {type: str, default: "off"}
tools:
  dynamic_setter:
    effects: [{set: mode, from: result.mode}]
  exact_setter:
    effects: [{set: mode, value: "on"}]
  needs_on:
    requires: [state.mode == "on"]
"""
    )
    plan = suggest_plan(policy, "needs_on", policy.initial_state(), {})
    assert plan.steps == ["exact_setter"]


def test_tool_edges(refund_policy):
    edges = tool_edges(refund_policy)
    assert ("verify_identity", "lookup_order", "state.identity_verified") in edges
    assert ("issue_refund", "send_confirmation_email", "after") in edges


def test_mermaid(refund_policy):
    full = to_mermaid(refund_policy)
    assert full.startswith("flowchart LR")
    assert "s_manager_approved" in full and ":::appstate" in full
    assert "t_issue_refund" in full
    assert "#quot;found#quot;" in full  # quotes are escaped
    md = to_markdown(refund_policy, tools_only=True)
    assert "```mermaid" in md


def test_cli_check_ok(policy_path, capsys):
    assert main(["check", str(policy_path)]) == 0
    assert "0 error(s)" in capsys.readouterr().out


def test_cli_check_fails_on_errors(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tools:\n  a:\n    after: [ghost]\n")
    assert main(["check", str(bad)]) == 1
    assert "unknown tool 'ghost'" in capsys.readouterr().out


def test_cli_explain(policy_path, capsys):
    assert main(["explain", str(policy_path), "issue_refund"]) == 0
    out = capsys.readouterr().out
    assert "Suggested order: verify_identity -> lookup_order -> issue_refund" in out
    main(["explain", str(policy_path), "lookup_order", "--state", "identity_verified=true"])
    assert "available right away" in capsys.readouterr().out


def test_cli_graph_and_schema(policy_path, tmp_path, capsys):
    out_file = tmp_path / "graph.md"
    assert main(["graph", str(policy_path), "--markdown", "-o", str(out_file)]) == 0
    assert "```mermaid" in out_file.read_text()
    capsys.readouterr()
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "stategate policy"
