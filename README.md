# parcus

**A local-first, token-thrift inference proxy for agentic harnesses** (Claude Code, pi,
opencode, …). Its sole purpose: **spend fewer tokens per turn** to preserve your budget, while
**preserving the meaning** the model needs to stay correct.

`pip install parcus` · runs on your machine · speaks Anthropic Messages + OpenAI Chat
Completions · everything beyond lossless compression and exact caching is **off by default** and
**fails open** (on any doubt it forwards your original request untouched).

---

## Why "parcus"?

**`parcus`** is Latin for *sparing, thrifty, frugal* — the literal root of the English word
**parsimony** (via *parcere*, "to spare"). The name *is* the mission in one word: **use no more
tokens than necessary.**

It carries two senses that both describe what the tool does:

- **Frugality** — be sparing with a scarce resource (your token budget); don't pay for what you
  don't need.
- **Parsimony / Occam's razor** (the "law of parsimony") — prefer the *minimal sufficient* form;
  strip away everything that doesn't carry meaning.

**Background.** Agentic harnesses re-send a lot of text every turn — system prompts, tool
schemas, conversation history, pasted context — and you pay for all of it, repeatedly. Most of
those tokens are either *redundant* (you already sent them) or *low-information* (filler that
doesn't change the model's behavior). `parcus` sits between your harness and the provider and
quietly removes that waste, without changing your results. The project was originally called
*parsimony*; that name is taken on PyPI, so it adopted its own Latin root — same idea, shorter.

---

## What it does

`parcus` is a transparent reverse proxy. Point your harness's base URL at it; it forwards to the
real Anthropic/OpenAI endpoint and, along the way, cuts tokens two ways:

1. **Compress the outbound request** — remove tokens that don't change the model's behavior, in
   safety **tiers**:
   - **Tier 0 — lossless** (on by default): whitespace/formatting normalization of prose only.
     Zero semantic risk.
   - **Tier 1 — filler removal** (opt-in): drop allow-listed discourse words ("please", "just",
     "obviously", …). A model-free guardrail proves *only* allow-listed whole tokens were
     removed.
   - **Tier 2 — learned** (opt-in, local model): drop low-information tokens with a local
     LLMLingua model; gated offline by an answer-preservation judge.

   Code, file paths, URLs, quoted text, tool JSON, and the trailing instruction are **never**
   altered.

2. **Avoid redundant calls** —
   - **Exact response cache**: identical requests replay the stored response byte-for-byte
     (prompts are never stored — only a salted hash).
   - **Semantic (near-duplicate) cache** (opt-in): serve a cached response when a new request is
     a near-duplicate, behind a high similarity threshold + a precision gate.
   - **Graph memory** (opt-in): retrieve only the relevant prior context / compact long
     histories instead of re-sending everything.

It also gives you **observability** (per-stage token reduction + accuracy, `stats`/JSON/Prometheus
endpoints), an optional **hosted/multi-tenant mode** (per-tenant isolation, edge authorization,
rate limiting), and optional **AES-256-GCM at-rest cache encryption** (with key rotation and
per-tenant keys / crypto-shredding).

## Design tenets

- **Fail open.** A token optimizer that breaks your harness or changes a result is worse than
  useless. On *any* uncertainty — unknown route, parse failure, compressor error — `parcus`
  forwards your **original, unmodified** request and returns the real response. (Security
  decisions still fail *closed*.)
- **Correctness is the gate; tokens are the objective.** No lossy transform ships unless it holds
  a measured no-regression bar.
- **Local-only.** Saving tokens by making *other* inference calls is self-defeating. Heuristics
  are model-free; optional compressors/embedders are **local** and never phone home.
- **Your keys and data stay yours.** Provider API keys are never logged, cached, or persisted.
  The proxy binds loopback/tailnet only — never the public internet.

## Quickstart

```bash
pip install parcus            # or: pip install -e ".[dev]" for development

# run the proxy (binds 127.0.0.1; refuses 0.0.0.0/public — set --host to a tailnet IP for
# other devices)
parcus serve --port 8787

# point your harness at it
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1

# measure token savings (+ correctness gate) over the built-in corpus or your own
parcus eval
parcus eval --filler            # evaluate the opt-in filler tier
parcus stats                    # aggregated per-stage reduction + accuracy
```

Configuration is via `PARCUS_*` environment variables (or a git-ignored `.env`) — see
[`.env.example`](.env.example) and the [Technical Reference](docs/technical-reference.md#7-configuration).

## Documentation

- **[The parcus guide](docs/guide.md)** — a plain-language book: the problem, the idea, and how
  to use each feature, for any reader.
- **[Technical reference](docs/technical-reference.md)** — the shapes of every piece: data model,
  ports, the request pipeline, each subsystem, the full config reference, HTTP surface, and CLI.
- **[FAQ](docs/faq.md)** — comprehensive answers, including correctness, privacy, and the name.
- **[Architecture decisions](docs/adr/)** (ADRs) and the **[threat model](docs/security/threat-model.md)**.
- **[PLAN.md](PLAN.md)** — original architecture & roadmap; **[CHANGELOG.md](CHANGELOG.md)** — release notes.

## Status

**v0.1.0** — the full tiered-compression pipeline, exact + semantic cache, graph memory,
observability, hosted multi-tenancy, and at-rest encryption are all implemented. CI runs the
security/quality gates (lint, type-check, SAST, tests with 100%-critical / ≥90% coverage, SCA,
SBOM, secret scan); releases are signed with SLSA provenance.

## Development

```bash
make setup     # venv + deps + pre-commit hooks
make test      # pytest + coverage gates
make lint      # ruff + mypy + bandit
make check     # everything CI runs
```

## License

[MIT](LICENSE).
