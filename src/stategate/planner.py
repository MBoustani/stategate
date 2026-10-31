"""The dependency graph and the planner.

Nobody writes the tool dependency graph by hand. It is *derived* from the policy:

* ``after: [a]`` on tool ``b`` gives an edge ``a -> b``.
* If tool ``b`` requires ``state.k == v`` and tool ``a`` has an effect that can
  make that true (``set: k, value: v``, or ``set: k, from: result.x``), then
  ``a -> b``.

When the guard blocks a call, the planner walks this graph backwards from the
blocked tool and suggests an order of tool calls that would unblock it. The
suggestion is only a *hint* for the agent. Enforcement is always the guard
re-checking the real state on every call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .conditions import Condition, Ref
from .policy import Policy


@dataclass(frozen=True)
class Provider:
    """A tool that can make a state condition true."""

    tool: str
    definite: bool  # True: a literal value that we know satisfies it. False: "may" (dynamic value).


def providers_for(policy: Policy, cond: Condition) -> list[Provider]:
    """Which tools have an effect that could make ``cond`` true?"""
    if cond.left.namespace != "state" or cond.left.key is None:
        return []
    key = cond.left.key
    simple = len(cond.left.path) == 1 and not isinstance(cond.right, Ref)
    found: list[Provider] = []
    for name, spec in policy.tools.items():
        best: Provider | None = None
        for effect in spec.effects:
            if effect.set != key:
                continue
            if effect.is_literal and simple:
                if cond.evaluate({"state": {key: effect.value}}):
                    best = Provider(name, True)
                    break
            elif effect.is_literal and effect.value is None:
                # Unsetting only helps "is not set" / "== null" style conditions.
                if cond.evaluate({"state": {}}):
                    best = best or Provider(name, False)
            else:
                best = best or Provider(name, False)
        if best is not None:
            found.append(best)
    # Prefer tools we *know* will satisfy the condition.
    found.sort(key=lambda p: not p.definite)
    return found


def tool_edges(policy: Policy) -> list[tuple[str, str, str]]:
    """Derived tool -> tool edges as ``(from_tool, to_tool, label)``."""
    edges: list[tuple[str, str, str]] = []
    for name, spec in policy.tools.items():
        for dep in spec.after:
            edges.append((dep, name, "after"))
        for cond in spec.state_conditions:
            for provider in providers_for(policy, cond):
                if provider.tool != name:
                    edges.append((provider.tool, name, str(cond.left)))
    # remove exact duplicates, keep order
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            unique.append(edge)
    return unique


@dataclass
class Plan:
    """A suggested order of tool calls, plus anything no tool can fix."""

    steps: list[str] = field(default_factory=list)
    unfixable: list[str] = field(default_factory=list)  # human-readable reasons


class _Hypothesis:
    """Imagined state while planning ("if the agent calls X, then Y becomes true")."""

    def __init__(self, state: Mapping[str, Any], calls: Mapping[str, int]) -> None:
        self.state = dict(state)
        self.assumed: set[str] = set()  # keys set from dynamic values; assume they work out
        self.done = {tool for tool, count in calls.items() if count > 0}
        self.calls = dict(calls)

    def copy(self) -> _Hypothesis:
        other = _Hypothesis(self.state, self.calls)
        other.assumed = set(self.assumed)
        other.done = set(self.done)
        return other

    def satisfied(self, cond: Condition) -> bool:
        if any(ref.namespace == "state" and ref.key in self.assumed for ref in cond.refs()):
            return True
        return cond.evaluate({"state": self.state})

    def apply(self, policy: Policy, tool: str) -> None:
        for effect in policy.tools[tool].effects:
            if effect.is_literal:
                self.state[effect.set] = effect.value
                self.assumed.discard(effect.set)
            else:
                self.assumed.add(effect.set)
        self.done.add(tool)
        self.calls[tool] = self.calls.get(tool, 0) + 1


def suggest_plan(
    policy: Policy,
    target: str,
    state: Mapping[str, Any],
    calls: Mapping[str, int],
) -> Plan:
    """Suggest the tool calls needed (in order) before ``target`` can run."""
    plan = Plan()

    def exhausted(tool: str, hypo: _Hypothesis) -> bool:
        limit = policy.tools[tool].max_calls
        return limit is not None and hypo.calls.get(tool, 0) >= limit

    def visit(tool: str, hypo: _Hypothesis, steps: list[str], stack: tuple[str, ...]) -> bool:
        """Try to make ``tool`` runnable; extend ``steps`` and ``hypo`` in place."""
        if tool in stack or tool not in policy.tools:
            return False
        stack = (*stack, tool)
        spec = policy.tools[tool]
        for dep in spec.after:
            if dep in hypo.done:
                continue
            if not _try(dep, hypo, steps, stack):
                return False
        for cond in spec.state_conditions:
            if hypo.satisfied(cond):
                continue
            ok = False
            for provider in providers_for(policy, cond):
                if provider.tool == tool:
                    continue
                if _try(provider.tool, hypo, steps, stack):
                    ok = hypo.satisfied(cond) or not provider.definite
                    if ok:
                        break
            if not ok:
                return False
        return True

    def _try(tool: str, hypo: _Hypothesis, steps: list[str], stack: tuple[str, ...]) -> bool:
        """Plan for ``tool`` and then 'call' it. Roll back if impossible."""
        if tool not in policy.tools or exhausted(tool, hypo):
            return False
        trial_hypo = hypo.copy()
        trial_steps = list(steps)
        if not visit(tool, trial_hypo, trial_steps, stack):
            return False
        trial_hypo.apply(policy, tool)
        trial_steps.append(tool)
        hypo.__dict__.update(trial_hypo.__dict__)
        steps[:] = trial_steps
        return True

    if target not in policy.tools:
        plan.unfixable.append(f"'{target}' is not listed in the policy")
        return plan

    hypo = _Hypothesis(state, calls)
    spec = policy.tools[target]
    steps: list[str] = []

    for dep in spec.after:
        if dep not in hypo.done and not _try(dep, hypo, steps, (target,)):
            plan.unfixable.append(f"no way found to complete required tool '{dep}' first")
    for cond in spec.state_conditions:
        if hypo.satisfied(cond):
            continue
        providers = [p for p in providers_for(policy, cond) if p.tool != target]
        if not providers:
            plan.unfixable.append(
                f"{cond}: no tool can change this. It is controlled by the application, not by the agent"
            )
            continue
        if not any(_try(p.tool, hypo, steps, (target,)) for p in providers):
            plan.unfixable.append(f"{cond}: the tools that could change it are blocked or used up")

    plan.steps = steps
    return plan
