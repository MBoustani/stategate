# Policy file reference

A policy is a YAML (or JSON) file with this shape:

```yaml
version: 1                  # required format version (always 1 for now)
name: my-agent              # optional
description: ...            # optional
unlisted_tools: deny        # deny (default) | allow

state:                      # the trusted state variables
  <name>: <state variable>

tools:                      # the rules for each tool
  <tool name>: <tool spec>
```

Unknown fields are rejected. A typo like `requirez:` is an error, not something silently ignored.

---

## `unlisted_tools`

What happens when the agent calls a tool that is **not** listed under `tools:`.

| Value | Meaning |
|---|---|
| `deny` (default) | The call is blocked. Recommended. |
| `allow` | The call is allowed (and still recorded in the audit log). Useful with the Claude `guard_hooks` when you only want to guard a few built-in tools. |

Every tool you want to allow must be listed. A tool with no rules can be written as `my_tool: {}`.

---

## `state`: trusted state variables

```yaml
state:
  identity_verified:
    type: bool            # bool | int | float | str | list | dict | any  (default: any)
    default: false        # starting value; leave out (or null) for "not set"
    enum: [a, b, c]       # optional: the only allowed values
    description: ...      # optional, shown to humans and to the agent
```

Shorthand: `identity_verified: bool` is the same as `identity_verified: {type: bool}`.

Rules:

- Names must be valid identifiers (`order_id`, not `order-id`) and cannot be `state`, `args` or `result`.
- Every write is type-checked. Storing a wrong type is refused, and the old value is kept.
- `null` means "not set". Any variable can be unset.
- `float` also accepts integers. `int` and `float` never accept `true`/`false`.

**Who can change state?** Only two things:

1. **Effects** of a tool call that succeeded (see below).
2. **Your application code**, via `Guard(policy, initial_state={...})` when the session starts, or `guard.set_state(key, value)` later (for example after a real login, or when a human approves something). Never expose `set_state` to the agent as a tool.

If no tool has an effect on a variable, only your application can set it. `stategate check` reports these as `info` so you can confirm that's intended.

> **YAML gotcha:** unquoted `yes`, `no`, `on`, `off` become `true`/`false` in YAML. Write `"on"` with quotes when you mean text. `stategate check` warns you when it sees this.

---

## `tools`: tool specs

```yaml
tools:
  issue_refund:
    description: ...                 # optional
    after: [lookup_order]            # tools that must have SUCCEEDED earlier in this session
    requires:                        # conditions that must ALL be true
      - state.identity_verified == true
      - args.amount <= $state.order_total
    max_calls: 1                     # max successful calls per session (0 = disabled)
    effects:                         # how state changes after success
      - set: order_status
        value: refunded
```

The tool name is what the agent calls. With the Claude Agent SDK hooks, `mcp__server__issue_refund` is matched to `issue_refund` automatically.

### `after`

A list of tool names. Each one must have completed successfully at least once in this session. A blocked or failed call doesn't count. Use it for pure ordering ("send the confirmation only after the refund"). When the order follows from *what happened*, prefer state requirements: they are more precise ("after a refund **of this order**").

### `requires`

A list of [conditions](#conditions). They can use:

- `state.*`: the trusted state. If one fails, the agent is told which tools could fix it, and the planner suggests an order.
- `args.*`: the arguments of **this call**. If one fails, the agent is told to fix its arguments. Argument rules are only reported after all state requirements are met.

They cannot use `result.*`, because the tool hasn't run yet.

### `max_calls`

Maximum number of **successful** calls per session (per `Guard`). Calls that are running count too, so parallel calls can't exceed the limit. `0` disables the tool completely.

### `effects`

Applied in order after a successful call, if the tool returned normally (no exception, and for Claude tools no `is_error`).

```yaml
effects:
  - set: identity_verified      # which state variable
    value: true                 # EITHER a literal value (null = unset) ...
  - set: order_id
    from: result.order_id       # ... OR copy from the call: result.<path> or args.<name>
  - set: order_status
    value: found
    when: result.found == true  # optional condition; may use result.*, args.*, state.*
```

- `value` and `from` are mutually exclusive, and one of them is required.
- `from` can only read `result.*` (what the tool returned) or `args.*` (what it was called with), never from other state.
- If `from` points to something that doesn't exist (`result.total` but the tool returned no `total`), the effect is skipped and an error is logged.
- All `when` conditions are evaluated against the state **before** this call's effects.

**What is `result`?** The tool's return value, normalized:

| Tool returns | `result` is |
|---|---|
| a dict | the dict: `result.order.id` walks into nested dicts |
| a pydantic model or dataclass | its fields as a dict |
| a JSON string like `'{"ok": true}'` | the parsed JSON |
| Claude MCP content `{"content": [{"type": "text", "text": "{...}"}]}` | the parsed JSON of the text |
| anything else | the value itself (use `result` alone) |

---

## Conditions

One condition per string:

```text
<ref> == | != | < | <= | > | >= <value>
<ref> in | not in <list>
<ref> is set
<ref> is not set
```

`<ref>` is `state.<name>`, `args.<name>` or `result[.<path>]`. Dots walk into nested dicts: `state.customer.tier`.

`<value>` can be:

| Written as | Meaning |
|---|---|
| `true`, `false`, `null` | booleans / null |
| `3`, `10.5` | numbers |
| `"found"` or `'found'` | text |
| `found` (bare word) | also text |
| `["a", "b"]` | a list (for `in` / `not in`) |
| `$state.order_total` | another value: compare two things |

YAML tips:

- Write `requires` as a block list (one `- ` per line). Conditions like `- state.status in ["a", "b"]` and `- state.status == "found"` work without extra quotes.
- The inline form `requires: [state.ok == true]` also works, but it **breaks** as soon as a condition contains its own `[...]` list. When in doubt, use the block form.
- A condition containing `: ` (colon + space) or starting with a special character such as `{` or `*` needs quotes around the whole line: `- 'state.note == "a: b"'`.

**Evaluation is fail-closed:**

- comparing with something unset gives `false` for `<`, `<=`, `>`, `>=`, `in`, `not in`
- `==` / `!=` treat unset as `null`
- comparing incompatible types (a number with text, a boolean with a number) gives `false`, never an error

---

## How the dependency graph is derived

There is no separate graph section. The graph follows from the rules:

- `after: [A]` on tool `B` gives `A ==> B`.
- `B` requires `state.k == v`, and `A` has an effect `set: k, value: v` gives `A --> B` (a definite provider).
- `B` requires something about `state.k`, and `A` has an effect `set: k, from: ...` gives `A --> B` (a possible provider, since the value is only known at runtime).

The planner walks this graph backwards from a blocked tool. It prefers definite providers and skips tools whose `max_calls` is used up. The result is the "Suggested order" the agent sees. The suggestion is only a hint: every call is checked against the real state again.

`stategate check` uses the same graph to find tools that can **never** run (for example `A` needs `B` and `B` needs `A`). That's an error, and `Guard` refuses to load such a policy.

---

## Python API cheat sheet

```python
from stategate import Guard, Policy

policy = Policy.from_file("policy.yaml")  # or Policy.from_yaml(text) / Policy.from_dict(d)
guard = Guard(policy, initial_state={"user_id": "u1"}, on_block="return")


@guard.tool  # or guard.wrap(fn, name="other_name")
def my_tool(x: int) -> dict: ...


guard.state["order_id"]  # read (read-only mapping)
guard.state.snapshot()  # dict copy
guard.set_state("manager_approved", True)  # trusted app code only
guard.check("my_tool", {"x": 1})  # Decision, no side effects
guard.status()  # state + which tools are available now
guard.audit_log()  # list of JSON-friendly events
guard.call_count("my_tool")

# Low-level lifecycle for custom integrations:
d = guard.begin("my_tool", args)
...
guard.finish("my_tool", args, result, call_id=d.call_id)
```

Logging: stategate logs decisions to the `stategate` logger. To see them:

```python
import logging

logging.basicConfig()
logging.getLogger("stategate").setLevel(logging.INFO)
```
