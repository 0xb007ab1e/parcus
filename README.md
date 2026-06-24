# parsimony

**A local-first, token-thrift inference proxy for agentic harnesses** (Claude Code, pi,
opencode, …). Its sole purpose: **reduce the tokens spent per turn** to preserve your budget,
while **preserving the semantic meaning** the model needs to stay correct.

> Status: **early bootstrap** (milestone M1 in progress — see [`PLAN.md`](PLAN.md)).

## How it works

`parsimony` sits transparently between your harness and the model provider. Point your
harness's base URL at the local proxy; it forwards to the real Anthropic / OpenAI endpoint,
and along the way:

1. **Compresses the outbound request** — removes tokens that don't change the model's
   behavior (tiered: always-safe lossless → opt-in, guardrailed filler removal → optional
   local learned compression). Code, paths, quoted text, and tool JSON are never altered.
2. **Avoids redundant calls** — an exact/normalized-hash response cache (and, later, graph
   memory for context retrieval & conversation compaction) so repeated context isn't paid for
   twice.

**Design tenets:** fail open (never break a harness — forward the original request on any
doubt), correctness over tokens (every lossy step is measured against a no-regression bar),
local-only models (never make another inference call to save one), confidential-by-default
cache (redact + TTL + opt-out).

## Quickstart

> Not yet runnable — bootstrap in progress. The intended flow:

```bash
# install (editable, with dev extras)
pip install -e ".[dev]"

# run the proxy (binds 127.0.0.1 + the host's tailnet IP, never public)
parsimony serve --port 8787

# point your harness at it
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

From a phone/other tailnet device: `http://<host>:8787` (MagicDNS name).

## Development

```bash
make setup     # venv + deps + pre-commit hooks
make test      # pytest + coverage gates
make lint      # ruff + mypy + bandit
make check     # everything CI runs
```

See [`PLAN.md`](PLAN.md) for architecture & roadmap, [`docs/`](docs/) for design/runbooks,
[`SECURITY.md`](SECURITY.md) for reporting, and `CLAUDE.md` for the engineering ruleset.

## License

[MIT](LICENSE).
