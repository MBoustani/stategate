# Changelog

## 0.1.0 (unreleased)

First version.

- Policy format (YAML/JSON) with trusted state variables, per-tool requirements, `after`, `max_calls` and effects
- Safe condition language (`state.x == true`, `args.amount <= $state.limit`, `in`, `is set`, ...)
- `Guard` runtime: blocks calls with an explanation and a suggested call order, applies effects, audit log, parallel-safe call limits
- Static checker (unknown names, type errors, circular dependencies, tools that can never run)
- Mermaid graph export, `stategate` CLI (`check`, `graph`, `explain`, `schema`)
- Adapters for the OpenAI Agents SDK and the Claude Agent SDK (custom tools and hooks for built-in tools)
