"""The condition language.

Conditions are tiny, safe expressions such as::

    state.identity_verified == true
    state.order_status in ["found", "shipped"]
    args.amount <= $state.order_total
    state.order_id is set

They are deliberately *not* Python code (no ``eval``). That keeps them safe to load
from a file, and simple enough that we can analyse them statically: the planner
and the checker can look at a condition and work out which tool could make it true.

Grammar (one condition per string)::

    <ref> == | != | < | <= | > | >= <operand>
    <ref> in | not in <operand>
    <ref> is set
    <ref> is not set

    <ref>     := state.<key>[.<field>...] | args.<name>[...] | result[.<field>...]
    <operand> := a JSON literal (true, 3.5, "text", ["a", "b"], null)
               | a bare word (treated as a string: found  ->  "found")
               | $<ref>  (compare against another value, e.g. $state.order_total)
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The places a condition can read values from.
#: ``state``  - the trusted state (only code can write it)
#: ``args``   - the arguments the agent passed to the tool call being checked
#: ``result`` - the value the tool returned (only available inside effects)
NAMESPACES = ("state", "args", "result")

BINARY_OPS = ("==", "!=", "<", "<=", ">", ">=")
MEMBERSHIP_OPS = ("in", "not in")
UNARY_OPS = ("is set", "is not set")
OPERATORS = BINARY_OPS + MEMBERSHIP_OPS + UNARY_OPS


class ConditionSyntaxError(ValueError):
    """Raised when a condition string cannot be parsed."""


class _Missing:
    """Sentinel for "this value does not exist" (different from ``None``)."""

    def __repr__(self) -> str:
        return "<unset>"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()


@dataclass(frozen=True)
class Ref:
    """A reference to a value, like ``state.order_id`` or ``result.order.total``."""

    namespace: str
    path: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> Ref:
        parts = text.strip().split(".")
        if parts[0] not in NAMESPACES:
            raise ConditionSyntaxError(f"'{text}' must start with one of: {', '.join(n + '.' for n in NAMESPACES)}")
        for part in parts[1:]:
            if not part.isidentifier():
                raise ConditionSyntaxError(f"'{part}' in '{text}' is not a valid name")
        if parts[0] != "result" and len(parts) < 2:
            raise ConditionSyntaxError(f"'{text}' needs a name after '{parts[0]}.'")
        return cls(parts[0], tuple(parts[1:]))

    @property
    def key(self) -> str | None:
        """The top-level name: ``state.order.id`` -> ``order``."""
        return self.path[0] if self.path else None

    def __str__(self) -> str:
        return ".".join((self.namespace, *self.path))


def resolve(ref: Ref, context: Mapping[str, Any]) -> Any:
    """Look up a reference in a context like ``{"state": {...}, "args": {...}}``.

    Returns :data:`MISSING` if any part of the path does not exist.
    """
    value = context.get(ref.namespace, MISSING)
    for part in ref.path:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return MISSING
    return value


def _parse_operand(text: str) -> Any:
    text = text.strip()
    if text.startswith("$"):
        return Ref.parse(text[1:])
    try:
        return json.loads(text)
    except ValueError:
        pass
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    if not text:
        raise ConditionSyntaxError("missing value on the right-hand side")
    return text  # a bare word is treated as a string


_NAME = r"[A-Za-z_][\w.]*"
_UNARY_RE = re.compile(rf"^\s*(?P<left>{_NAME})\s+(?P<op>is\s+not\s+set|is\s+set)\s*$")
_MEMBER_RE = re.compile(rf"^\s*(?P<left>{_NAME})\s+(?P<op>not\s+in|in)\s+(?P<right>.+?)\s*$")
_BINARY_RE = re.compile(rf"^\s*(?P<left>{_NAME})\s*(?P<op>==|!=|<=|>=|<|>)\s*(?P<right>.+?)\s*$")


def format_value(value: Any) -> str:
    """Render a value the way it would be written in a policy file."""
    if value is MISSING:
        return "<unset>"
    if isinstance(value, Ref):
        return f"${value}"
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)


@dataclass(frozen=True)
class Condition:
    """A parsed condition. Create one with :meth:`Condition.parse`."""

    left: Ref
    op: str
    right: Any = None  # a literal value, or a Ref
    source: str = ""

    @classmethod
    def parse(cls, text: str) -> Condition:
        if not isinstance(text, str):
            raise ConditionSyntaxError(f"a condition must be a string, got {type(text).__name__}")
        for regex in (_UNARY_RE, _MEMBER_RE, _BINARY_RE):
            match = regex.match(text)
            if match:
                break
        else:
            raise ConditionSyntaxError(
                f"cannot understand condition '{text}'. Expected something like "
                "'state.verified == true', 'args.amount <= $state.limit', "
                "'state.status in [\"a\", \"b\"]' or 'state.order_id is set'"
            )
        op = " ".join(match.group("op").split())
        left = Ref.parse(match.group("left"))
        right: Any = None
        if op not in UNARY_OPS:
            right = _parse_operand(match.group("right"))
            if op in MEMBERSHIP_OPS and not isinstance(right, (list, Ref)):
                raise ConditionSyntaxError(f"'{op}' needs a list on the right, e.g. [\"a\", \"b\"] (in '{text}')")
        return cls(left=left, op=op, right=right, source=text.strip())

    # ------------------------------------------------------------------ helpers
    def refs(self) -> list[Ref]:
        """All references this condition reads (left side and, if any, right side)."""
        out = [self.left]
        if isinstance(self.right, Ref):
            out.append(self.right)
        return out

    def namespaces(self) -> set[str]:
        return {ref.namespace for ref in self.refs()}

    def __str__(self) -> str:
        if self.source:
            return self.source
        if self.op in UNARY_OPS:
            return f"{self.left} {self.op}"
        return f"{self.left} {self.op} {format_value(self.right)}"

    # --------------------------------------------------------------- evaluation
    def evaluate(self, context: Mapping[str, Any]) -> bool:
        """Return True if the condition holds. Never raises: anything odd is False.

        "Odd" includes comparing incompatible types or ordering against an unset
        value. Returning False means the guard *blocks* - we fail closed.
        """
        left = resolve(self.left, context)
        if self.op == "is set":
            return left is not MISSING and left is not None
        if self.op == "is not set":
            return left is MISSING or left is None

        right = resolve(self.right, context) if isinstance(self.right, Ref) else self.right
        lv = None if left is MISSING else left
        rv = None if right is MISSING else right
        try:
            if self.op == "==":
                return bool(lv == rv)
            if self.op == "!=":
                return bool(lv != rv)
            if self.op == "in":
                return rv is not None and lv in rv
            if self.op == "not in":
                return rv is not None and lv not in rv
            if lv is None or rv is None:
                return False
            if isinstance(lv, bool) != isinstance(rv, bool):
                return False  # don't let True < 5 sneak through
            if self.op == "<":
                return bool(lv < rv)
            if self.op == "<=":
                return bool(lv <= rv)
            if self.op == ">":
                return bool(lv > rv)
            if self.op == ">=":
                return bool(lv >= rv)
        except TypeError:
            return False
        return False  # pragma: no cover - unknown operator is impossible after parse

    def explain(self, context: Mapping[str, Any]) -> str:
        """Human/LLM friendly description including the current values involved."""
        parts = [f"{ref} is {format_value(resolve(ref, context))}" for ref in self.refs()]
        return f"{self}  (currently: {', '.join(parts)})"
