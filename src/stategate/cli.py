"""Command line interface.

stategate check   policy.yaml            # find mistakes (exit code 1 on errors)
stategate graph   policy.yaml            # Mermaid diagram of the dependency graph
stategate explain policy.yaml TOOL       # what does TOOL need, and in what order?
stategate schema                         # JSON Schema for policy files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .checker import check_policy
from .guard import Guard
from .policy import Policy, PolicyError
from .render import describe_tool, to_markdown, to_mermaid


def _parse_assignment(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got '{text}'")
    key, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    return key.strip(), value


def _load(path: str) -> Policy:
    try:
        return Policy.from_file(path)
    except FileNotFoundError:
        sys.exit(f"error: file not found: {path}")
    except PolicyError as exc:
        sys.exit(f"error: could not load {path}:\n{exc}")


def _cmd_check(args: argparse.Namespace) -> int:
    policy = _load(args.policy)
    issues = check_policy(policy)
    if not issues:
        print(f"OK: {args.policy} - {len(policy.state)} state variable(s), {len(policy.tools)} tool(s), no issues.")
        return 0
    for issue in issues:
        print(issue)
    errors = sum(i.severity == "error" for i in issues)
    warnings = sum(i.severity == "warning" for i in issues)
    print(f"\n{errors} error(s), {warnings} warning(s)")
    if errors or (args.strict and warnings):
        return 1
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    policy = _load(args.policy)
    text = (
        to_markdown(policy, tools_only=args.tools_only)
        if args.markdown
        else to_mermaid(policy, tools_only=args.tools_only)
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    policy = _load(args.policy)
    if args.tool not in policy.tools:
        sys.exit(f"error: '{args.tool}' is not in the policy. Tools: {', '.join(policy.tools) or '(none)'}")
    print(describe_tool(policy, args.tool))
    try:
        guard = Guard(policy, initial_state=dict(args.state or []), validate=False)
    except PolicyError as exc:
        sys.exit(f"error: {exc}")
    decision = guard.check(args.tool)
    print()
    if decision.allowed:
        print("With this state the tool is available right away (argument rules are still checked per call).")
    else:
        print("Starting from this state, the agent would see:\n")
        print(decision.message())
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    text = json.dumps(Policy.json_schema(), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stategate",
        description="Deterministic tool guardrails for AI agents: trusted state + tool dependency graph.",
    )
    parser.add_argument("--version", action="version", version=f"stategate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="validate a policy file")
    p.add_argument("policy")
    p.add_argument("--strict", action="store_true", help="also fail on warnings")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("graph", help="print the dependency graph as Mermaid")
    p.add_argument("policy")
    p.add_argument("--tools-only", action="store_true", help="only show tool -> tool edges")
    p.add_argument("--markdown", action="store_true", help="wrap in a Markdown file with a legend")
    p.add_argument("-o", "--output", help="write to a file instead of stdout")
    p.set_defaults(func=_cmd_graph)

    p = sub.add_parser("explain", help="show what a tool needs and the suggested call order")
    p.add_argument("policy")
    p.add_argument("tool")
    p.add_argument(
        "--state",
        action="append",
        type=_parse_assignment,
        metavar="KEY=VALUE",
        help="start from this state instead of the defaults (repeatable), e.g. --state identity_verified=true",
    )
    p.set_defaults(func=_cmd_explain)

    p = sub.add_parser("schema", help="print the JSON Schema for policy files")
    p.add_argument("-o", "--output", help="write to a file instead of stdout")
    p.set_defaults(func=_cmd_schema)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
