"""Static checks for a policy - run before the agent ever starts.

This is the "human review" helper: ``stategate check policy.yaml`` lists
mistakes (typos in state names, wrong value types) and design problems
(a tool that can never run because of a circular dependency).

Severity levels:

* ``error``   - the policy is broken; :class:`stategate.Guard` refuses to load it.
* ``warning`` - probably a mistake; review it.
* ``info``    - good to know (e.g. "this state is controlled by your application").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .conditions import Condition, Ref
from .planner import providers_for
from .policy import Policy

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.where}: {self.message}"


_YAML_BOOL_HINT = " (note: in YAML, unquoted yes/no/on/off/true/false become booleans - put quotes around text values)"


def _hint(var_type: str, value: object) -> str:
    return _YAML_BOOL_HINT if var_type == "str" and isinstance(value, bool) else ""


def _state_refs(cond: Condition) -> list[Ref]:
    return [r for r in cond.refs() if r.namespace == "state"]


def check_policy(policy: Policy) -> list[Issue]:
    """Return all problems found in ``policy`` (an empty list means all good)."""
    issues: list[Issue] = []

    def add(severity: Severity, where: str, message: str) -> None:
        issues.append(Issue(severity, where, message))

    # --- state declarations -------------------------------------------------
    for key, var in policy.state.items():
        if var.enum is not None:
            probe = var.model_copy(update={"enum": None})
            bad = [v for v in var.enum if not probe.accepts(v)]
            if bad:
                add("error", f"state.{key}", f"enum values {bad} are not of type {var.type}")
        if not var.accepts(var.default):
            add(
                "error",
                f"state.{key}",
                f"default {var.default!r} is not {var.describe_type()}" + _hint(var.type, var.default),
            )

    def check_state_ref(ref: Ref, where: str) -> bool:
        if ref.key not in policy.state:
            add("error", where, f"'{ref}' refers to unknown state variable '{ref.key}'")
            return False
        return True

    def check_literal(cond: Condition, where: str) -> None:
        """Compare the literal on the right with the declared type of the left."""
        if cond.left.namespace != "state" or len(cond.left.path) != 1:
            return
        var = policy.state.get(cond.left.key or "")
        if var is None or isinstance(cond.right, Ref) or cond.op in ("is set", "is not set"):
            return
        values = cond.right if cond.op in ("in", "not in") else [cond.right]
        for value in values:
            if value is not None and not var.accepts(value):
                add(
                    "error",
                    where,
                    f"'{cond}' compares state.{cond.left.key} ({var.describe_type()}) with {value!r} - it can never match"
                    + _hint(var.type, value),
                )

    # --- tools --------------------------------------------------------------
    for name, spec in policy.tools.items():
        for dep in spec.after:
            if dep == name:
                add("error", f"tools.{name}.after", "a tool cannot depend on itself")
            elif dep not in policy.tools:
                add("error", f"tools.{name}.after", f"unknown tool '{dep}'")
        for cond in spec.conditions:
            where = f"tools.{name}.requires"
            if all(check_state_ref(ref, where) for ref in _state_refs(cond)):
                check_literal(cond, where)
        for i, effect in enumerate(spec.effects):
            where = f"tools.{name}.effects[{i}]"
            var = policy.state.get(effect.set)
            if var is None:
                add("error", where, f"sets unknown state variable '{effect.set}'")
            elif effect.is_literal and not var.accepts(effect.value):
                add(
                    "error",
                    where,
                    f"value {effect.value!r} is not {var.describe_type()} (state.{effect.set})"
                    + _hint(var.type, effect.value),
                )
            if effect.when_condition is not None:
                for ref in _state_refs(effect.when_condition):
                    check_state_ref(ref, where + ".when")
        if spec.max_calls == 0:
            add("warning", f"tools.{name}", "max_calls is 0, so this tool can never run")

    if any(i.severity == "error" for i in issues):
        return issues  # the analysis below assumes references are valid

    # --- who controls each state variable ---------------------------------------
    read_keys = {ref.key for spec in policy.tools.values() for cond in spec.conditions for ref in _state_refs(cond)}
    for key in policy.state:
        writers = policy.writers_of(key)
        if key in read_keys and not writers:
            add(
                "info",
                f"state.{key}",
                "no tool changes this variable, so only your application code can set it "
                "(Guard(initial_state=...) or guard.set_state(...))",
            )
        if key not in read_keys and not writers:
            add("warning", f"state.{key}", "declared but never used by any tool")

    # --- reachability: can every tool ever run? ---------------------------------
    # Optimistic fixed point: a tool is reachable if each requirement is already
    # true by default, or can be made true by a reachable tool, or is controlled
    # by the application (no tool writes it).
    defaults = {"state": policy.initial_state()}
    reachable: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, spec in policy.tools.items():
            if name in reachable or spec.max_calls == 0:
                continue
            if any(dep not in reachable for dep in spec.after):
                continue
            ok = True
            for cond in spec.state_conditions:
                if cond.evaluate(defaults):
                    continue
                keys = {ref.key for ref in _state_refs(cond)}
                if any(not policy.writers_of(k) for k in keys):
                    continue  # application-controlled
                if isinstance(cond.right, Ref):
                    continue  # compares two values; can't judge statically
                providers = [p.tool for p in providers_for(policy, cond) if p.tool != name]
                if not providers:
                    add(
                        "warning",
                        f"tools.{name}.requires",
                        f"no tool can make '{cond}' true (only the application can)",
                    )
                    continue
                if not any(p in reachable for p in providers):
                    ok = False
                    break
            if ok:
                reachable.add(name)
                changed = True

    for name, spec in policy.tools.items():
        if name not in reachable and spec.max_calls != 0:
            add(
                "error",
                f"tools.{name}",
                "can never run: its requirements depend on tools that can never run "
                "(look for a circular dependency, e.g. A needs B and B needs A)",
            )

    # Deduplicate (the fixed point may report the same warning more than once).
    unique: list[Issue] = []
    for issue in issues:
        if issue not in unique:
            unique.append(issue)
    return unique
