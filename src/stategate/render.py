"""Draw the dependency graph as a Mermaid diagram.

Mermaid is a text format for diagrams. GitHub, GitLab, Notion and many docs tools
render it automatically, so ``stategate graph policy.yaml --markdown > GRAPH.md``
gives you a picture your reviewers can look at without installing anything.
"""

from __future__ import annotations

import re

from .planner import providers_for, tool_edges
from .policy import Policy


def _node_id(prefix: str, name: str) -> str:
    return prefix + re.sub(r"\W", "_", name)


def _label(text: str) -> str:
    """Escape text for a quoted Mermaid label."""
    return text.replace('"', "#quot;").replace("\n", " ")


def to_mermaid(policy: Policy, *, tools_only: bool = False) -> str:
    """Return Mermaid ``flowchart`` source for the policy.

    * Full view (default): tools are boxes, state variables are cylinders.
      ``tool --sets--> state`` and ``state --required by--> tool``.
    * ``tools_only=True``: just tools, with an arrow ``A --> B`` when A unlocks B.
    """
    lines = ["flowchart LR"]
    lines.append("  classDef tool fill:#e8f0fe,stroke:#3367d6,color:#111;")
    lines.append("  classDef state fill:#fff4d6,stroke:#c98a00,color:#111;")
    lines.append("  classDef appstate fill:#fde2e1,stroke:#c5221f,color:#111;")

    for name, spec in policy.tools.items():
        extra = f"<br/><small>max {spec.max_calls} call(s)</small>" if spec.max_calls is not None else ""
        lines.append(f'  {_node_id("t_", name)}["{_label(name)}{extra}"]:::tool')

    if tools_only:
        for src, dst, label in tool_edges(policy):
            style = "==>" if label == "after" else "-->"
            lines.append(f'  {_node_id("t_", src)} {style}|"{_label(label)}"| {_node_id("t_", dst)}')
        return "\n".join(lines) + "\n"

    for key in policy.state:
        cls = "state" if policy.writers_of(key) else "appstate"
        lines.append(f'  {_node_id("s_", key)}[("{_label(key)}")]:::{cls}')

    for name, spec in policy.tools.items():
        for effect in spec.effects:
            if effect.set in policy.state:
                lines.append(
                    f'  {_node_id("t_", name)} -->|"sets {_label(effect.describe().split(" = ", 1)[1])}"| '
                    f"{_node_id('s_', effect.set)}"
                )
        for dep in spec.after:
            lines.append(f'  {_node_id("t_", dep)} ==>|"after"| {_node_id("t_", name)}')
        for cond in spec.conditions:
            for ref in cond.refs():
                if ref.namespace == "state" and ref.key in policy.state:
                    lines.append(f'  {_node_id("s_", ref.key)} -.->|"{_label(str(cond))}"| {_node_id("t_", name)}')
    return "\n".join(lines) + "\n"


def to_markdown(policy: Policy, *, tools_only: bool = False) -> str:
    """Mermaid diagram wrapped in a Markdown code block, plus a legend."""
    title = policy.name or "stategate policy"
    legend = (
        "Legend: blue boxes are tools, yellow cylinders are state set by tools, "
        "red cylinders are state only your application can set. "
        "Dotted arrows are requirements, thick arrows are `after` dependencies."
    )
    if tools_only:
        legend = "Legend: an arrow A --> B means calling A can unlock B (the label is the state involved)."
    return f"# {title}\n\n{legend}\n\n```mermaid\n{to_mermaid(policy, tools_only=tools_only)}```\n"


def describe_tool(policy: Policy, tool: str) -> str:
    """Plain-text summary of one tool's rules (used by ``stategate explain``)."""
    spec = policy.tools[tool]
    out = [f"Tool: {tool}"]
    if spec.description:
        out.append(f"  {spec.description}")
    if spec.after:
        out.append("  Must run after: " + ", ".join(spec.after))
    if spec.conditions:
        out.append("  Requires:")
        for cond in spec.conditions:
            fixers = [p.tool for p in providers_for(policy, cond) if p.tool != tool]
            hint = f"   <- set by: {', '.join(fixers)}" if fixers else ""
            if "args" in cond.namespaces():
                hint = "   <- about the call's arguments"
            out.append(f"    - {cond}{hint}")
    if spec.max_calls is not None:
        out.append(f"  Max calls per session: {spec.max_calls}")
    if spec.effects:
        out.append("  On success:")
        out.extend(f"    - {e.describe()}" for e in spec.effects)
    return "\n".join(out)
