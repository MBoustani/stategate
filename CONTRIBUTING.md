# Contributing

Thanks for your interest in stategate!

## Getting set up

```bash
git clone https://github.com/MBoustani/stategate
cd stategate
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,openai,claude]"
pytest
ruff check .
```

The adapter tests run against the real `openai-agents` and `claude-agent-sdk` packages. They don't call any model API, so no API keys are needed. If those packages aren't installed, the adapter tests are skipped.

## Guidelines

- **Fail closed.** If something is unknown, missing or the wrong type, the answer is "blocked". New features should keep that property.
- **Keep the core small.** The core only depends on `pydantic` and `pyyaml`. Framework-specific code goes in `src/stategate/adapters/` and must import its framework lazily.
- **Messages are part of the product.** The text the agent sees when a call is blocked should say what is wrong and what to do next.
- **Add a test** for every bug fix and feature.

## Project layout

```text
src/stategate/
  conditions.py   the condition language (parse + evaluate)
  policy.py       the policy schema (pydantic models, YAML/JSON loading)
  state.py        the trusted state store (read-only to everyone but the Guard)
  planner.py      the derived dependency graph + "suggested order" planner
  checker.py      static checks for human review
  guard.py        the runtime: check, block/allow, apply effects, audit log
  render.py       Mermaid diagrams
  cli.py          the `stategate` command
  adapters/       OpenAI Agents SDK and Claude Agent SDK integrations
```

## Reporting security issues

If you find a way to make a guarded tool run when its requirements are not met, please report it privately (see the repository's Security tab) instead of opening a public issue.
