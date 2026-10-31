"""The trusted state store.

The agent can *read* this state, but it can never write it. Values only change when:

1. a tool call succeeds and the policy has an ``effects`` rule for it, or
2. your own application code calls :meth:`stategate.Guard.set_state`
   (for example right after a real login).

:class:`TrustedState` itself is a read-only mapping, so you can safely hand it to
other code (``state["order_id"]``, ``dict(state)``, ``"x" in state``).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Any

from .policy import Policy, PolicyError


class StateTypeError(PolicyError):
    """Raised when someone tries to store a value of the wrong type."""


class TrustedState(Mapping[str, Any]):
    """Read-only view of the trusted state. Writes go through :class:`stategate.Guard`."""

    def __init__(self, policy: Policy, initial: Mapping[str, Any] | None = None) -> None:
        self._policy = policy
        self._values: dict[str, Any] = dict(policy.initial_state())
        for key, value in (initial or {}).items():
            self._write(key, value)

    # Mapping interface (read-only) -------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if key not in self._policy.state:
            raise KeyError(key)
        return self._values.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._policy.state)

    def __len__(self) -> int:
        return len(self._policy.state)

    def __repr__(self) -> str:
        return f"TrustedState({self.snapshot()!r})"

    def snapshot(self) -> dict[str, Any]:
        """A deep copy of all variables (unset ones are ``None``)."""
        return {key: copy.deepcopy(self._values.get(key)) for key in self._policy.state}

    def is_set(self, key: str) -> bool:
        return self._values.get(key) is not None

    # Internal write path - only the Guard calls this --------------------------
    def _write(self, key: str, value: Any) -> tuple[Any, Any]:
        var = self._policy.state.get(key)
        if var is None:
            raise StateTypeError(f"unknown state variable '{key}' (declare it under 'state:' in the policy)")
        if not var.accepts(value):
            raise StateTypeError(
                f"state variable '{key}' must be {var.describe_type()}, got {value!r} ({type(value).__name__})"
            )
        old = self._values.get(key)
        if value is None:
            self._values.pop(key, None)
        else:
            self._values[key] = copy.deepcopy(value)
        return old, value
