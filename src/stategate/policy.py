"""The policy schema: what state exists, and what each tool requires and changes.

A policy is usually written in YAML (or JSON) and loaded with
:meth:`Policy.from_file`. It has two main sections:

``state``
    The trusted state variables, declared up front with a type and a default.
    Only code can change them - never the agent.

``tools``
    For each tool: what must be true before it may run (``requires`` and
    ``after``), how often it may run (``max_calls``), and how the trusted state
    changes when it succeeds (``effects``).

The tool dependency graph is *derived* from this: if ``lookup_order`` requires
``state.identity_verified == true`` and ``verify_identity`` has an effect that
sets ``identity_verified`` to ``true``, then ``verify_identity -> lookup_order``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

from .conditions import NAMESPACES, Condition, ConditionSyntaxError, Ref

StateType = Literal["bool", "int", "float", "str", "list", "dict", "any"]

_PY_TYPES: dict[str, tuple[type, ...]] = {
    "bool": (bool,),
    "int": (int,),
    "float": (int, float),
    "str": (str,),
    "list": (list, tuple),
    "dict": (dict,),
}


class PolicyError(ValueError):
    """Raised when a policy cannot be loaded or has errors."""


class StateVar(BaseModel):
    """Declaration of one trusted state variable."""

    model_config = ConfigDict(extra="forbid")

    type: StateType = Field("any", description="Value type. Values of the wrong type are rejected.")
    default: Any = Field(None, description="Initial value. null means 'not set'.")
    enum: list[Any] | None = Field(None, description="If given, the only allowed values.")
    description: str = Field("", description="What this variable means. Shown to humans and to the agent.")

    def accepts(self, value: Any) -> bool:
        """True if ``value`` is allowed for this variable (``None`` = unset is always allowed)."""
        if value is None:
            return True
        if self.type != "any":
            allowed = _PY_TYPES[self.type]
            if not isinstance(value, allowed):
                return False
            if self.type in ("int", "float") and isinstance(value, bool):
                return False  # in Python, True is an int - we don't want that here
        if self.enum is not None and value not in self.enum:
            return False
        return True

    def describe_type(self) -> str:
        if self.enum is not None:
            return f"one of {json.dumps(self.enum)}"
        return self.type


class Effect(BaseModel):
    """How the trusted state changes after a tool call succeeds.

    Examples (YAML)::

        - set: identity_verified        # literal value
          value: true
          when: result.verified == true # only if the tool's result says so

        - set: order_total              # copy a value out of the tool's result
          from: result.total
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    set: str = Field(..., description="Name of the state variable to change.")
    value: Any = Field(None, description="Literal value to store. Use null to unset.")
    from_: str | None = Field(
        None,
        alias="from",
        description="Copy the value from the tool call: 'result.<field>' or 'args.<name>'.",
    )
    when: str | None = Field(
        None,
        description="Optional condition (may use result.*, args.*, state.*). The effect only applies if it holds.",
    )

    _from_ref: Ref | None = PrivateAttr(None)
    _when: Condition | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _validate(self) -> Effect:
        has_value = "value" in self.model_fields_set
        if has_value and self.from_ is not None:
            raise ValueError(f"effect on '{self.set}': use either 'value' or 'from', not both")
        if not has_value and self.from_ is None:
            raise ValueError(f"effect on '{self.set}': needs 'value' or 'from'")
        if self.from_ is not None:
            try:
                ref = Ref.parse(self.from_)
            except ConditionSyntaxError as exc:
                raise ValueError(f"effect on '{self.set}': bad 'from': {exc}") from exc
            if ref.namespace not in ("result", "args"):
                raise ValueError(
                    f"effect on '{self.set}': 'from' must start with 'result' or 'args.' "
                    "(state is only changed from what tools actually did)"
                )
            self._from_ref = ref
        if self.when is not None:
            try:
                self._when = Condition.parse(self.when)
            except ConditionSyntaxError as exc:
                raise ValueError(f"effect on '{self.set}': bad 'when': {exc}") from exc
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        # Only write the fields that were meant: `value` OR `from`, and `when` if any.
        data = handler(self)
        if self.from_ is not None:
            data.pop("value", None)
        else:
            data.pop("from", None)
            data.pop("from_", None)
            data["value"] = self.value  # keep `value: null` even with exclude_defaults
        if self.when is None:
            data.pop("when", None)
        return data

    @property
    def is_literal(self) -> bool:
        return self.from_ is None

    @property
    def from_ref(self) -> Ref | None:
        return self._from_ref

    @property
    def when_condition(self) -> Condition | None:
        return self._when

    def describe(self) -> str:
        source = json.dumps(self.value) if self.is_literal else self.from_
        text = f"{self.set} = {source}"
        if self.when:
            text += f"  (when {self.when})"
        return text


class ToolSpec(BaseModel):
    """The rules for one tool."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field("", description="Optional note for humans.")
    after: list[str] = Field(
        default_factory=list,
        description="Tools that must have completed successfully (at least once) before this one.",
    )
    requires: list[str] = Field(
        default_factory=list,
        description="Conditions on state.* and args.* that must ALL hold before the tool may run.",
    )
    max_calls: int | None = Field(
        None, ge=0, description="Maximum number of successful calls per session. 0 disables the tool."
    )
    effects: list[Effect] = Field(
        default_factory=list, description="How the trusted state changes when the tool succeeds."
    )

    _requires: list[Condition] = PrivateAttr(default_factory=list)

    @field_validator("requires", mode="before")
    @classmethod
    def _one_string_is_a_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("after", mode="before")
    @classmethod
    def _after_string_is_a_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _parse_requires(self) -> ToolSpec:
        parsed = []
        for text in self.requires:
            try:
                cond = Condition.parse(text)
            except ConditionSyntaxError as exc:
                raise ValueError(f"bad requirement: {exc}") from exc
            if "result" in cond.namespaces():
                raise ValueError(
                    f"requirement '{text}' uses 'result', but the tool has not run yet. "
                    "Use state.* or args.* in requirements."
                )
            parsed.append(cond)
        self._requires = parsed
        return self

    @property
    def conditions(self) -> list[Condition]:
        return list(self._requires)

    @property
    def state_conditions(self) -> list[Condition]:
        """Requirements that only look at state (these can be planned for)."""
        return [c for c in self._requires if "args" not in c.namespaces()]

    @property
    def args_conditions(self) -> list[Condition]:
        """Requirements that look at the call's arguments (the agent must fix its call)."""
        return [c for c in self._requires if "args" in c.namespaces()]


class Policy(BaseModel):
    """A complete stategate policy. See ``docs/policy-reference.md`` for the format."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = Field(1, description="Policy format version.")
    name: str = Field("", description="Optional name of this policy.")
    description: str = Field("", description="Optional description.")
    unlisted_tools: Literal["deny", "allow"] = Field(
        "deny",
        description="What to do with a tool that is not listed under 'tools'. 'deny' is the safe default.",
    )
    state: dict[str, StateVar] = Field(default_factory=dict, description="Trusted state variables.")
    tools: dict[str, ToolSpec] = Field(default_factory=dict, description="Rules per tool.")

    @field_validator("state", mode="before")
    @classmethod
    def _shorthand_state(cls, value: Any) -> Any:
        # Allow the shorthand  `identity_verified: bool`  instead of `{type: bool}`.
        if isinstance(value, dict):
            return {k: ({"type": v} if isinstance(v, str) else v) for k, v in value.items()}
        return value

    @field_validator("tools", mode="before")
    @classmethod
    def _empty_tools(cls, value: Any) -> Any:
        # Allow `some_tool:` with nothing under it (YAML gives None) = no rules.
        if isinstance(value, dict):
            return {k: ({} if v is None else v) for k, v in value.items()}
        return value

    @field_validator("state")
    @classmethod
    def _check_state_names(cls, value: dict[str, StateVar]) -> dict[str, StateVar]:
        for key in value:
            if not key.isidentifier():
                raise ValueError(f"state name '{key}' must be a valid identifier (letters, digits, _)")
            if key in NAMESPACES:
                raise ValueError(f"'{key}' is a reserved word and cannot be a state name")
        return value

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        try:
            return cls.model_validate(data)
        except ValueError as exc:
            raise PolicyError(str(exc)) from exc

    @classmethod
    def from_yaml(cls, text: str) -> Policy:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyError(f"invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise PolicyError("a policy file must contain a mapping at the top level")
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        """Load a policy from a ``.yaml``/``.yml`` or ``.json`` file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            try:
                return cls.from_dict(json.loads(text))
            except json.JSONDecodeError as exc:
                raise PolicyError(f"invalid JSON in {path}: {exc}") from exc
        return cls.from_yaml(text)

    def to_yaml(self) -> str:
        data = self.model_dump(by_alias=True, exclude_defaults=True)
        return yaml.safe_dump(data, sort_keys=False)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """JSON Schema for policy files (useful for editor autocompletion and for LLM drafting)."""
        schema = cls.model_json_schema(by_alias=True)
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = "stategate policy"
        return schema

    # ---------------------------------------------------------------- queries
    def initial_state(self) -> dict[str, Any]:
        return {key: var.default for key, var in self.state.items() if var.default is not None}

    def writers_of(self, key: str) -> list[str]:
        """Tools that have at least one effect on state variable ``key``."""
        return [name for name, spec in self.tools.items() if any(e.set == key for e in spec.effects)]
