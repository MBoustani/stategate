"""The runtime guard.

One :class:`Guard` = one agent session (one conversation). It owns:

* the trusted state for that session,
* a count of how many times each tool succeeded,
* an audit log of every decision.

Typical use::

    policy = Policy.from_file("policy.yaml")
    guard = Guard(policy)

    @guard.tool
    def issue_refund(order_id: str, amount: float) -> dict:
        ...

When the agent calls ``issue_refund`` the guard checks the policy *before* your
function runs. If a requirement is not met, your function is **not called**, and
the agent gets back a message explaining what is missing and which tools to call
first. If the call is allowed and succeeds, the policy's ``effects`` update the
trusted state.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from .conditions import MISSING, resolve
from .planner import Plan, providers_for, suggest_plan
from .policy import Policy, PolicyError
from .state import StateTypeError, TrustedState

logger = logging.getLogger("stategate")

F = TypeVar("F", bound=Callable[..., Any])

UnmetKind = Literal["unlisted", "disabled", "max_calls", "after", "state", "args"]


@dataclass
class Unmet:
    """One reason a tool call was blocked."""

    kind: UnmetKind
    message: str
    fixable_by: list[str] = field(default_factory=list)  # tools that could fix it


@dataclass
class Decision:
    """The guard's answer for one tool call."""

    tool: str
    allowed: bool
    unmet: list[Unmet] = field(default_factory=list)
    plan: Plan | None = None
    call_id: str | None = None

    def message(self) -> str:
        """The text sent back to the agent when a call is blocked."""
        if self.allowed:
            return f"`{self.tool}` is allowed."
        lines = [f"BLOCKED: `{self.tool}` was not executed because its requirements are not met."]
        hard = [u for u in self.unmet if u.kind in ("unlisted", "disabled", "max_calls")]
        if hard:
            # Nothing the agent does can fix these, so don't send it on a detour.
            lines.extend(f"  - {u.message}" for u in hard)
            lines.append("This cannot be fixed by calling other tools. Do not retry this tool; continue without it.")
            return "\n".join(lines)
        lines.append("Unmet requirements:")
        for i, item in enumerate(self.unmet, 1):
            line = f"  {i}. {item.message}"
            if item.fixable_by:
                line += f"\n     -> can be satisfied by calling: {', '.join(item.fixable_by)}"
            lines.append(line)
        if self.plan and self.plan.steps:
            lines.append("Suggested order: " + " -> ".join([*self.plan.steps, self.tool]))
        if self.plan and self.plan.unfixable:
            lines.append("These cannot be fixed by calling tools:")
            lines.extend(f"  - {reason}" for reason in self.plan.unfixable)
        if any(item.kind == "args" for item in self.unmet):
            lines.append("Fix the arguments of your call so they match the requirements above.")
        lines.append(
            "Note: the trusted state is managed by the system. You cannot change it directly; "
            "it is updated only when the required tools succeed."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "allowed": self.allowed,
            "unmet": [dataclasses.asdict(u) for u in self.unmet],
            "suggested_order": [*(self.plan.steps if self.plan else []), self.tool] if not self.allowed else [],
        }


class ToolBlocked(Exception):
    """Raised instead of returning a message when ``on_block="raise"``."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.message())
        self.decision = decision


@dataclass
class StateChange:
    key: str
    old: Any
    new: Any
    source: str  # "tool:<name>" or "app"


@dataclass
class AuditEvent:
    timestamp: float
    event: Literal["blocked", "succeeded", "failed", "state_set"]
    tool: str | None
    args: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    changes: list[StateChange] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "tool": self.tool,
            "args": _json_safe(self.args),
            "reasons": list(self.reasons),
            "changes": [
                {"key": c.key, "old": _json_safe(c.old), "new": _json_safe(c.new), "source": c.source}
                for c in self.changes
            ],
        }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return repr(value)


def normalize_result(result: Any) -> Any:
    """Turn a tool's return value into something ``result.<field>`` can read."""
    if isinstance(result, Mapping):
        return dict(result)
    model_dump = getattr(result, "model_dump", None)  # pydantic models
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:  # pragma: no cover - defensive
            return result
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    if isinstance(result, str):
        text = result.strip()
        if text[:1] in ("{", "["):
            try:
                return json.loads(text)
            except ValueError:
                pass
    return result


class Guard:
    """Enforces a :class:`Policy` for one agent session.

    Args:
        policy: the policy to enforce.
        initial_state: values to start with (overrides the policy defaults).
            Use this for facts your application already knows, e.g. the logged-in user.
        on_block: ``"return"`` (default) returns the explanation as the tool's
            result so the agent can read it and recover. ``"raise"`` raises
            :class:`ToolBlocked` instead.
        validate: run the static checker on creation and refuse policies with errors.
    """

    def __init__(
        self,
        policy: Policy,
        *,
        initial_state: Mapping[str, Any] | None = None,
        on_block: Literal["return", "raise"] = "return",
        validate: bool = True,
    ) -> None:
        if validate:
            from .checker import check_policy

            errors = [i for i in check_policy(policy) if i.severity == "error"]
            if errors:
                details = "\n".join(f"  - {e}" for e in errors)
                raise PolicyError(f"the policy has {len(errors)} error(s):\n{details}")
        self.policy = policy
        self.on_block = on_block
        self._state = TrustedState(policy, initial_state)
        self._calls: Counter[str] = Counter()
        self._inflight: dict[str, str] = {}  # call_id -> tool
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._audit: list[AuditEvent] = []
        self.managed_tools: set[str] = set()  # tools wrapped by stategate itself

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> TrustedState:
        """Read-only view of the trusted state."""
        return self._state

    def set_state(self, key: str, value: Any, *, source: str = "app") -> None:
        """Change the trusted state from **your own code** (never expose this to the agent).

        Use this for facts the application establishes itself, e.g. after a real
        login: ``guard.set_state("user_authenticated", True)``.
        """
        with self._lock:
            old, new = self._state._write(key, value)
            change = StateChange(key, old, new, source)
            self._audit.append(AuditEvent(time.time(), "state_set", None, {}, changes=[change]))
        logger.info("state %s: %r -> %r (%s)", key, old, new, source)

    def call_count(self, tool: str) -> int:
        """How many times ``tool`` has completed successfully in this session."""
        return self._calls[tool]

    # --------------------------------------------------------------- checking
    def _context(self, args: Mapping[str, Any] | None) -> dict[str, Any]:
        return {"state": self._state.snapshot(), "args": dict(args or {})}

    def _evaluate(self, tool: str, args: Mapping[str, Any] | None, *, include_args: bool) -> Decision:
        spec = self.policy.tools.get(tool)
        if spec is None:
            if self.policy.unlisted_tools == "allow":
                return Decision(tool, True)
            return Decision(
                tool,
                False,
                [
                    Unmet(
                        "unlisted",
                        f"`{tool}` is not allowed by the policy (it is not listed and unlisted tools are denied)",
                    )
                ],
            )

        ctx = self._context(args)
        unmet: list[Unmet] = []

        if spec.max_calls is not None:
            used = self._calls[tool] + sum(1 for t in self._inflight.values() if t == tool)
            if spec.max_calls == 0:
                unmet.append(Unmet("disabled", f"`{tool}` is disabled by the policy (max_calls: 0)"))
            elif used >= spec.max_calls:
                unmet.append(
                    Unmet(
                        "max_calls",
                        f"`{tool}` may only be called {spec.max_calls} time(s) per session and that limit is reached",
                    )
                )

        for dep in spec.after:
            if self._calls[dep] == 0:
                unmet.append(Unmet("after", f"`{dep}` must complete successfully before `{tool}`", [dep]))

        for cond in spec.state_conditions:
            if not cond.evaluate(ctx):
                fixers = [p.tool for p in providers_for(self.policy, cond) if p.tool != tool]
                unmet.append(Unmet("state", cond.explain(ctx), fixers))

        # Argument rules are only reported once the state is ready; before that they
        # usually compare against unset values and would just confuse the agent.
        if include_args and not unmet:
            for cond in spec.args_conditions:
                if not cond.evaluate(ctx):
                    unmet.append(Unmet("args", cond.explain(ctx)))

        decision = Decision(tool, not unmet, unmet)
        if unmet and any(u.kind in ("state", "after") for u in unmet):
            decision.plan = suggest_plan(self.policy, tool, ctx["state"], dict(self._calls))
        return decision

    def check(self, tool: str, args: Mapping[str, Any] | None = None) -> Decision:
        """Would ``tool`` be allowed with these ``args`` right now? (No side effects.)

        If ``args`` is None, argument requirements are skipped.
        """
        with self._lock:
            return self._evaluate(tool, args, include_args=args is not None)

    # ------------------------------------------------------- call lifecycle
    # These three methods are what the adapters use. ``wrap`` uses them too.
    def begin(self, tool: str, args: Mapping[str, Any], *, call_id: str | None = None) -> Decision:
        """Check a call and, if allowed, reserve it (so parallel calls can't exceed limits)."""
        with self._lock:
            decision = self._evaluate(tool, args, include_args=True)
            if decision.allowed:
                decision.call_id = call_id or f"call-{next(self._ids)}"
                self._inflight[decision.call_id] = tool
            else:
                self._audit.append(
                    AuditEvent(time.time(), "blocked", tool, dict(args), [u.message for u in decision.unmet])
                )
        if not decision.allowed:
            logger.warning("blocked %s: %s", tool, "; ".join(u.message for u in decision.unmet))
        return decision

    def finish(
        self, tool: str, args: Mapping[str, Any], result: Any, *, call_id: str | None = None
    ) -> list[StateChange]:
        """Record a successful call and apply its effects to the trusted state."""
        with self._lock:
            if call_id is not None:
                self._inflight.pop(call_id, None)
            changes: list[StateChange] = []
            spec = self.policy.tools.get(tool)
            if spec is not None:
                ctx = {"state": self._state.snapshot(), "args": dict(args), "result": normalize_result(result)}
                for effect in spec.effects:
                    when = effect.when_condition
                    if when is not None and not when.evaluate(ctx):
                        continue
                    if effect.is_literal:
                        value = effect.value
                    else:
                        assert effect.from_ref is not None
                        value = resolve(effect.from_ref, ctx)
                        if value is MISSING:
                            logger.error(
                                "effect %s of tool %s skipped: %s not found in the call",
                                effect.set,
                                tool,
                                effect.from_,
                            )
                            continue
                    try:
                        old, new = self._state._write(effect.set, value)
                    except StateTypeError as exc:
                        # Fail closed: keep the old value, the requirement stays unmet.
                        logger.error("effect of tool %s skipped: %s", tool, exc)
                        continue
                    changes.append(StateChange(effect.set, old, new, f"tool:{tool}"))
            self._calls[tool] += 1
            self._audit.append(AuditEvent(time.time(), "succeeded", tool, dict(args), changes=changes))
        for c in changes:
            logger.info("state %s: %r -> %r (%s)", c.key, c.old, c.new, c.source)
        return changes

    def abort(
        self, tool: str, args: Mapping[str, Any], error: BaseException | str, *, call_id: str | None = None
    ) -> None:
        """Record a failed call. No effects are applied and it does not count as completed."""
        with self._lock:
            if call_id is not None:
                self._inflight.pop(call_id, None)
            self._audit.append(
                AuditEvent(time.time(), "failed", tool, dict(args), [str(error) or type(error).__name__])
            )

    # --------------------------------------------------------------- wrapping
    def wrap(self, fn: F, *, name: str | None = None, skip_params: tuple[str, ...] = ()) -> F:
        """Return a guarded version of ``fn`` (sync or async).

        ``name`` is the tool name used in the policy (defaults to ``fn.__name__``).
        ``skip_params`` are parameter names that are not tool arguments (e.g. a
        framework context object); they are left out of ``args``.
        """
        tool = name or fn.__name__
        if tool not in self.policy.tools and self.policy.unlisted_tools == "deny":
            raise PolicyError(
                f"tool '{tool}' is not listed in the policy, and unlisted tools are denied. "
                f"Add it under 'tools:' (it can be empty: `{tool}: {{}}`)."
            )
        self.managed_tools.add(tool)
        sig = inspect.signature(fn)

        def bind(a: tuple[Any, ...], kw: dict[str, Any]) -> dict[str, Any]:
            try:
                bound = sig.bind(*a, **kw)
                bound.apply_defaults()
                values = dict(bound.arguments)
            except TypeError:
                values = dict(kw)
            for skip in skip_params:
                values.pop(skip, None)
            return values

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*a: Any, **kw: Any) -> Any:
                args = bind(a, kw)
                decision = self.begin(tool, args)
                if not decision.allowed:
                    return self._blocked(decision)
                try:
                    result = await fn(*a, **kw)
                except BaseException as exc:
                    self.abort(tool, args, exc, call_id=decision.call_id)
                    raise
                self.finish(tool, args, result, call_id=decision.call_id)
                return result

            async_wrapper.__stategate_tool__ = tool  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            args = bind(a, kw)
            decision = self.begin(tool, args)
            if not decision.allowed:
                return self._blocked(decision)
            try:
                result = fn(*a, **kw)
            except BaseException as exc:
                self.abort(tool, args, exc, call_id=decision.call_id)
                raise
            self.finish(tool, args, result, call_id=decision.call_id)
            return result

        wrapper.__stategate_tool__ = tool  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    def tool(self, fn: F | None = None, *, name: str | None = None) -> Any:
        """Decorator form of :meth:`wrap`: ``@guard.tool`` or ``@guard.tool(name="x")``."""
        if fn is None:
            return lambda f: self.wrap(f, name=name)
        return self.wrap(fn, name=name)

    def _blocked(self, decision: Decision) -> str:
        if self.on_block == "raise":
            raise ToolBlocked(decision)
        return decision.message()

    # ------------------------------------------------------------ reporting
    def status(self) -> dict[str, Any]:
        """State plus which tools are currently available. Safe to show to the agent."""
        with self._lock:
            tools: dict[str, Any] = {}
            for name in self.policy.tools:
                decision = self._evaluate(name, None, include_args=False)
                entry: dict[str, Any] = {"available_now": decision.allowed, "completed_calls": self._calls[name]}
                if not decision.allowed:
                    entry["waiting_for"] = [u.message for u in decision.unmet]
                if self.policy.tools[name].args_conditions:
                    entry["argument_rules"] = [str(c) for c in self.policy.tools[name].args_conditions]
                tools[name] = entry
            state = {
                key: {"value": _json_safe(self._state[key]), "description": var.description}
                for key, var in self.policy.state.items()
            }
            return {"state": state, "tools": tools}

    def status_text(self) -> str:
        """:meth:`status` as pretty JSON, ready to return from a read-only tool."""
        return json.dumps(self.status(), indent=2)

    @property
    def audit(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._audit)

    def audit_log(self) -> list[dict[str, Any]]:
        """The audit log as JSON-friendly dicts."""
        return [event.to_dict() for event in self.audit]


__all__ = [
    "AuditEvent",
    "Decision",
    "Guard",
    "StateChange",
    "ToolBlocked",
    "Unmet",
    "normalize_result",
]
