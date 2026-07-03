# parcus

**A local-first, token-thrift inference proxy for agentic harnesses** — it spends fewer tokens
per turn to preserve your budget, while preserving the meaning the model needs to stay correct.

`pip install parcus` · runs entirely on your machine · speaks **Anthropic Messages** + **OpenAI
Chat Completions** · transparent (point a base URL at it and forget it's there) · everything
beyond lossless compression and exact caching is **off by default** and **fails open** — on any
doubt it forwards your original request, untouched.

---

## Why "parcus"?

**`parcus`** is Latin for *sparing, thrifty, frugal* — the literal root of the English word
**parsimony** (via *parcere*, "to spare"). The name *is* the mission in one word: **use no more
tokens than necessary.**

It carries two senses that both describe exactly what the tool does:

- **Frugality** — be sparing with a scarce, metered resource (your token budget); never pay for
  what you don't need.
- **Parsimony / Occam's razor** — the *law of parsimony*: prefer the **minimal sufficient** form;
  *entia non sunt multiplicanda praeter necessitatem* ("entities must not be multiplied beyond
  necessity"). Swap "entities" for "tokens" and you have the product specification.

**Background.** Agentic harnesses — Claude Code, pi, opencode, Cursor-style agents, home-grown
tool-using loops — are remarkably token-hungry. On **every single turn** they re-send a large,
mostly-unchanged payload: a long system prompt, the full set of tool/function schemas, the entire
conversation so far, and whatever files or context you've pasted in. The model is stateless
between calls, so the harness keeps resending everything, and you are billed for all of it,
repeatedly, for the whole session. Most of those tokens are either **redundant** (you already
sent that exact context last turn) or **low-information** (politeness and filler that doesn't
change the model's behavior). `parcus` sits between your harness and the provider and quietly
removes that waste — without changing your results. *(The project was originally named*
parsimony*; that name is taken on PyPI, so it adopted its own Latin root — same idea, shorter.)*

## What is parcus?

`parcus` is a small **reverse proxy** you run on your own computer (or your private tailnet). You
point your harness's API base URL at the local proxy; it forwards each request to the real
provider endpoint and streams the real response straight back. In between, on the requests it can
safely understand, it does two complementary things:

1. **Makes each request smaller** — it rewrites only the *prose* parts of your prompt to say the
   same thing in fewer tokens, in escalating safety tiers (see below). Code, file paths, URLs,
   quoted text, tool/function JSON, and your trailing instruction are treated as **immutable** and
   reproduced byte-for-byte.
2. **Avoids paying for the same answer twice** — it remembers responses and replays them for
   identical (and, optionally, near-identical) requests instead of calling the provider again, and
   can keep a compact memory of your session so the harness needn't re-send everything each turn.

It is **not** a model, a router to cheaper models, or a cloud service. It runs locally, makes no
inference calls of its own, and is designed to be **invisible**: if it ever can't confidently
improve a request, it gets out of the way and forwards the original.

## How it works (the request pipeline)

For each request parcus can parse, it: detects the provider dialect → (in hosted mode) derives
the tenant from the credential and applies authorization + rate limits → parses to a single
provider-agnostic canonical model → optionally compacts via memory → runs the compression chain →
checks the response cache (exact, then optional near-duplicate) → forwards what remains upstream →
stores a cacheable response. **Every** optimization step fails open: any error or ambiguity yields
the original, unmodified request and the real response. Outcomes are surfaced on `x-parcus-*`
response headers (`cache`, `dialect`, `tokens-before/after/saved`).

## Features

### Tiered request compression
Compression is layered from zero-risk to more aggressive, so you choose exactly how much you
trust:

- **Tier 0 — lossless** *(on by default)*: whitespace/formatting normalization of prose spans
  only. Cannot change meaning; verified by a model-free invariant on every request.
- **Tier 1 — filler removal** *(opt-in)*: drops whole tokens from a curated allow-list of
  discourse fillers ("please", "just", "obviously", …). A **model-free guardrail** proves that
  *only* allow-listed tokens were removed — nothing else is dropped, added, or reordered. Ships a
  conservative default set and a larger aggressive set.
- **Tier 2 — learned** *(opt-in, local model)*: drops low-information tokens with a local
  LLMLingua model. This tier is genuinely lossy/semantic, so it has no runtime proof and is
  instead gated **offline** by an answer-preservation judge before you enable it.

### Response caching
- **Exact cache** *(on by default)*: a byte-for-byte replay when an identical request recurs.
  Prompts are **never stored** — only a salted hash — and credential-bearing or pattern-excluded
  requests are never cached.
- **Semantic / near-duplicate cache** *(opt-in)*: serve a cached response when a new request is a
  *near*-duplicate (same model and tenant, cosine similarity above a deliberately high threshold),
  validated by a no-false-hit precision gate. Uses a **local** embedder; the safe semantic model
  is the default.

### Graph memory *(opt-in)*
Keeps a graph of durable facts/decisions from your session and either injects only the *relevant*
subset (context retrieval) or replaces older turns with a rolling summary (conversation
compaction) — so the harness stops re-sending everything. Behind a retrieval-recall gate, because
it changes what the model sees.

### Observability
Per-stage token reduction **and** accuracy (the model-free invariant pass-rate) are recorded for
every turn — content-free counts only, never prompt text. Read them via `parcus stats`, a JSON
endpoint, or Prometheus; a health endpoint is included.

### Hosted / multi-tenant mode *(opt-in)*
Run one shared instance for a team with strict per-tenant isolation: the tenant is derived
**server-side** from the inbound credential (never a client-supplied field), each tenant's cache
and memory are isolated, an optional allow-list authorizes callers at the edge, and per-tenant
rate limits prevent noisy-neighbor abuse.

### At-rest cache encryption *(opt-in)*
Encrypt cached response bodies with **AES-256-GCM** (authenticated; tamper-evident), with the key
from your environment/keyfile (never committed), graceful key rotation, and — in hosted mode —
per-tenant derived keys with **crypto-shredding** (withhold a tenant's key to erase their cached
data instantly).

### Correctness tooling
A built-in eval harness measures token reduction *and* enforces a no-regression bar: model-free
equivalence for the safe tiers, and quality/precision gates for the lossy ones. Run `parcus eval`
(and `--filler`, `--retrieval`, `--similarity`) against the built-in corpus or your own dataset.

## Design tenets

- **Fail open.** A token optimizer that breaks your harness or changes a result is worse than
  useless. On any uncertainty, parcus forwards your **original, unmodified** request. (Security
  decisions still fail *closed*.)
- **Correctness is the gate; tokens are the objective.** No lossy transform ships unless it holds
  a measured no-regression bar.
- **Local-only.** Saving tokens by making *other* inference calls is self-defeating; the optional
  compressors/embedders are local and never phone home.
- **Your keys and data stay yours.** Provider API keys are never logged, cached, or persisted; the
  proxy binds loopback/tailnet only — never the public internet.

## Is parcus for you?

Yes, if you run token-hungry agentic tools and want to cut spend and latency without risking your
results — especially long sessions with big system prompts, many tools, and lots of repeated
context. It's single-user and local by default; the hosted mode adds the controls for a shared
team instance. It is **not** for changing which model you use or for any setup where you can't run
a local process between your harness and the provider.

## Quickstart

```bash
pip install parcus

# run the proxy (binds 127.0.0.1; refuses 0.0.0.0/public — set --host to a tailnet IP for others)
parcus serve --port 8787

# point your harness at it
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1

# measure token savings (+ the correctness gate) over the built-in corpus or your own dataset
parcus eval
parcus eval --filler          # evaluate the opt-in filler tier
parcus stats                  # aggregated per-stage reduction + accuracy
```

Configuration is via `PARCUS_*` environment variables (or a git-ignored `.env`) — see
[`.env.example`](https://github.com/0xb007ab1e/parcus/blob/main/.env.example) and the
[configuration reference](https://github.com/0xb007ab1e/parcus/blob/main/docs/technical-reference.md#7-configuration).
Everything beyond Tier-0 + exact cache is opt-in; a safe progression is `FILLER` → (validate) →
`FILLER_AGGRESSIVE`/`LEARNED` → `SIMILARITY_CACHE` → `MEMORY`.

## Documentation

- **[The parcus guide](https://github.com/0xb007ab1e/parcus/blob/main/docs/guide.md)** — a
  plain-language book: the problem, the idea, and how to use each feature, for any reader.
- **[Technical reference](https://github.com/0xb007ab1e/parcus/blob/main/docs/technical-reference.md)** —
  the shapes of every piece: data model, ports, the request pipeline, each subsystem, the full
  config reference, the HTTP surface, and the CLI.
- **[FAQ](https://github.com/0xb007ab1e/parcus/blob/main/docs/faq.md)** — correctness, privacy,
  the name, and more.
- **[Architecture decisions (ADRs)](https://github.com/0xb007ab1e/parcus/tree/main/docs/adr)** and
  the **[threat model](https://github.com/0xb007ab1e/parcus/blob/main/docs/security/threat-model.md)**.
- **[CHANGELOG](https://github.com/0xb007ab1e/parcus/blob/main/CHANGELOG.md)**.

## Status

**v0.2.0** ([release notes](https://github.com/0xb007ab1e/parcus/releases/tag/v0.2.0)) — the full
tiered-compression pipeline, exact + semantic cache, graph memory, observability, hosted
multi-tenancy, and at-rest encryption are all implemented and tested. **0.2.0** adds a **KMS-backed
master key** and **irreversible per-tenant crypto-shredding** for the at-rest cache, an
**answer-preservation gate** for the lossy tiers (`parcus eval --learned`), fail-open hardening
across every adapter seam, and a broad test-depth expansion (property / fault-injection /
streaming / concurrency / credential-safety suites, plus dependency-free load/fuzz/DAST/mutation
harnesses). All backward-compatible; new features are opt-in. CI enforces the security/quality
gates (lint, strict typing, SAST, tests with 100%-critical-path / ≥90% coverage, dependency audit,
SBOM, secret scan, Markdown link check); releases are signed with SLSA build provenance.

## Development

```bash
make setup     # venv + deps + pre-commit hooks
make test      # pytest + coverage gates
make lint      # ruff + mypy + bandit
make check     # everything CI runs
```

### Optional: codebase-memory index

parcus can be indexed into a codebase-memory knowledge graph (an MCP server) for structural code
navigation (who-calls-what, dead-code, impact tracing). This is an **optional dev aid** — it is
**not** a dependency of parcus and is not needed to build, test, or run the project.

If you have the codebase-memory MCP server installed, build a local graph with one command:

```bash
/codebase-memory index      # via Claude Code, with the MCP server connected
```

The generated index lives in `.codebase-memory/` and is **git-ignored** (a ~1.4 MB binary that
changes on every re-index — regenerate it locally rather than vendoring it). Contributors without
the MCP server can ignore this section entirely.

**When the memory MCP is available and the project has been indexed, prefer it for tracking the
project shape** — call graphs (who-calls-what), impact/blast-radius of a change, and dead-code
detection — over ad-hoc `grep`/`glob`. It answers structural questions precisely and cheaply; keep
the index fresh by re-running `/codebase-memory index` after significant structural changes.

## License

[MIT](https://github.com/0xb007ab1e/parcus/blob/main/LICENSE).
