# parsimony — Plan

A **local-first, token-thrift inference proxy** for agentic harnesses (Claude Code, pi,
opencode, …). Its sole purpose is to **reduce tokens spent per turn** — preserving the end
user's budget — while **preserving the semantic meaning** the model needs to do the task
correctly. Correctness is the gate; token reduction is the objective.

> Inherits the master SSDLC ruleset (`~/.claude/CLAUDE.md`) and the project `CLAUDE.md`.

---

## 1. Problem & approach

Agentic harnesses re-send large, repetitive context every turn (system prompts, tool
schemas, file contents, prior turns). Two levers reduce spend:

1. **Compress the outbound request** — strip tokens that don't change the model's behavior.
2. **Avoid the call entirely** — serve a known answer / inject only the relevant memory
   instead of round-tripping the whole context to the provider.

We sit as a **transparent proxy** between the harness and the provider so *every* harness
benefits with **zero per-harness code**. The harness points its base URL at
`http://127.0.0.1:<port>` (and the host's tailnet IP for phone/other devices); we forward to
the real provider.

### Non-negotiable design tenets
- **Fail open.** If anything is uncertain — unknown route, parse failure, cache miss,
  compressor error — forward the **original, unmodified** request upstream. The proxy must
  never break a harness or change a result to save tokens.
- **Correctness > tokens.** Every lossy transform is measured against a quality bar on an
  eval set; a regression fails the gate (master §4).
- **Local-only models.** Saving tokens by making *other* inference calls is self-defeating.
  Heuristics are model-free; any embedding/learned compressor is a **local** model, lazy and
  opt-in.
- **Confidential by default.** Prompts/responses may contain secrets/PII → redact before
  persist, TTL expiry, opt-out patterns, kill switch (master §5, `std-privacy`).

---

## 2. Locked scope decisions (from planning Q&A)

| Area | Decision |
|---|---|
| Integration | Transparent inference **proxy/gateway** |
| API surfaces (day 1) | **Anthropic Messages**, **OpenAI Chat Completions**, **transparent pass-through** |
| Stack | **Python** (3.12+) |
| Compression | **Hybrid tiered**: always-on lossless + opt-in lossy filler removal w/ guardrails |
| Graph memory | **Explore all three**: response cache · context-retrieval graph · conversation compaction |
| Cache policy | **Exact/normalized-hash only** by default; embedding-similarity opt-in, off by default |
| Deployment | **Local-first** (loopback + tailnet), **hosted-ready** behind interfaces |
| Success metric | **Max % token reduction at equal quality** (no-regression gate) |
| Local models | **Local-only, lazy/optional**; never an outbound call to save tokens |
| Storage | **SQLite + sqlite-vec** embedded; networkx for graph algorithms |
| First milestone | **Working proxy: lossless + hash cache + eval harness** |
| Cache privacy | **Redact + TTL + opt-out**, classified confidential |

### Explicitly out of scope (for now)
- Multi-tenant hosted service, auth/billing (architecture stays clean for it — not built).
- Aggressive semantic cache that returns "close enough" answers (correctness risk).
- Fine-tuning / training any model.
- Modifying provider responses (we only ever compress *requests* and serve *exact* cache hits).

---

## 3. Architecture (ports & adapters / functional core)

Business logic is pure and provider-agnostic; I/O lives at the edges (`topic-architecture-patterns`,
`topic-dependency-injection`). This is what keeps it hosted-ready without building hosting now.

```
        harness                         parsimony proxy (imperative shell)                    provider
   ┌──────────────┐   HTTP    ┌───────────────────────────────────────────────┐   HTTP   ┌────────────┐
   │ Claude Code  │ ───────▶  │  ingress (FastAPI)                             │ ──────▶  │ Anthropic  │
   │ pi / opencode│           │   │ dialect-detect → canonical request model   │          │ OpenAI     │
   └──────────────┘  ◀─────── │   ▼                                            │ ◀──────  │ …          │
                              │  pipeline (pure functional core):              │          └────────────┘
                              │    Redactor → Compressor* → CacheKey           │
                              │    → CacheLookup → [hit? short-circuit]        │
                              │    → MemoryContext (retrieval/compaction)      │
                              │  ports (adapters injected at composition root):│
                              │    CompressorPort  CachePort  MemoryPort       │
                              │    StorePort       TokenizerPort  RedactorPort │
                              │    UpstreamPort     ClockPort                  │
                              └───────────────────────────────────────────────┘
   *Compressor = tiered: LosslessPass (always) ∘ FillerPass (opt-in, guardrailed)
```

### Canonical request model
Each provider dialect is parsed into one internal `CanonicalRequest` (messages, system,
tools, params, stream flag, provenance of which spans are *immutable* — code, quoted text,
tool JSON). Compression operates on the canonical model; we re-serialize to the original
dialect on the way out. Unknown dialect → no canonicalization → pass-through.

### Ports (interfaces)
- `TokenizerPort` — count tokens per provider/model (drives every measurement).
- `RedactorPort` — detect+mask secrets/PII spans.
- `CompressorPort` — `compress(canonical) -> (canonical, Stats)`; composable passes.
- `CachePort` — `get(key)`, `put(key, value, ttl)`; exact/normalized hash.
- `MemoryPort` — graph memory: `ingest(turn)`, `relevant(context) -> spans`.
- `StorePort` — SQLite/sqlite-vec backing (cache + graph share it).
- `UpstreamPort` — forward to real provider (streaming + non-streaming).
- `ClockPort` — injected time (TTL, testability — `topic-numeric-correctness`).

Adapters are wired once in a **composition root**; the core never imports FastAPI, httpx,
or sqlite directly.

---

## 4. Compression: tiered & guardrailed

The model's behavior must not change. We classify every span as **immutable** or
**reducible** before touching anything.

**Immutable (never altered):** fenced/inline code, file paths, URLs, quoted strings, tool
schema JSON, structured data (tables/JSON/YAML), numbers/IDs, the trailing user instruction.

**Tier 0 — Lossless (default ON, zero semantic risk)**
- Collapse redundant whitespace/blank lines outside code.
- De-duplicate verbatim repeated blocks (e.g. the same file pasted twice) → reference.
- Strip zero-information boilerplate (trailing "Let me know if…", redundant restatement).
- Prompt-cache-aware ordering hints (keep stable prefix stable so the *provider's* cache hits;
  ties to `topic-token-optimization`).

**Tier 1 — Filler removal (opt-in, guardrailed, lossy)**
- Remove discourse fillers / low-information function words in *prose* spans only, via
  POS-aware heuristics — never in immutable spans.
- Bounded aggressiveness setting; each transform is reversible-logged for audit.
- **Guardrail:** every Tier-1 ruleset is validated on the eval set; ship only configs that
  hold the no-regression bar.

**Tier 2 — Learned (opt-in, local model)**
- LLMLingua-2-style local token-importance scoring for prose. Local model only; lazy-loaded;
  off unless enabled. Heaviest savings, measured hardest.

Each tier emits `Stats` (tokens before/after, spans touched) for the eval harness.

---

## 5. Graph memory — three tracks (research)

Shared substrate: a property graph persisted in SQLite (nodes/edges + sqlite-vec embeddings),
queried with networkx for algorithms. All three are **read-augmenting**; only Track A can
skip a call, and only on an **exact/normalized** key by default.

- **Track A — Response cache.** Normalized-hash key over the canonical request → stored
  response. Exact-only default; tool-result dedup. Optional embedding-similarity is a
  separate, off-by-default mode with a strict threshold and a "verify before serve" hook.
- **Track B — Context-retrieval graph (RAG-lite).** Build a graph of durable project facts /
  decisions / entities from the conversation + files; inject only the relevant subgraph
  instead of re-sending big context each turn. Reduces *input* tokens.
- **Track C — Conversation compaction.** Maintain an entity/state graph across turns; replace
  verbose history with a compact, derived summary while preserving referenced facts.
  Reduces *history* tokens.

Research deliverable: measure each track's token saving vs. quality cost on the eval set;
keep the winners.

---

## 6. Milestones

- **M1 (first shippable):** proxy (Anthropic + OpenAI + pass-through) · Tier-0 lossless ·
  exact-hash cache + redaction · streaming passthrough · eval/measurement harness ·
  fail-open everywhere. **Real token savings, low risk.**
- **M2:** Tier-1 filler removal with the guardrail pipeline; CLI + config; observability/metrics
  dashboard of savings.
- **M3:** Graph memory Track A hardening + Track B (context retrieval) behind the `MemoryPort`.
- **M4:** Track C (compaction); optional Tier-2 learned compressor (local); embedding-similarity
  cache mode (opt-in).
- **M5:** Hosted-ready hardening (multi-tenancy, authn) — only if pursued.

Each milestone is "done" only at the master §8 bar (tests/gates/docs/threat-model-as-needed).

---

## 7. Success metrics & research questions

**Headline:** % input-token reduction per turn at **equal task quality** (no-regression on the
eval set). Secondary: outbound-call reduction (cache hit rate), added per-call latency
(must stay small — local, fast), blended $/task.

**Open research questions**
1. How much lossless-only saving is achievable on real agentic traffic (system prompts +
   tool schemas + file context)?
2. Which filler-removal rules save tokens without measurably changing agent behavior?
3. Does provider-prompt-cache-aware ordering beat naive compression (don't break their cache)?
4. For graph memory: retrieval-graph (B) vs compaction (C) — which saves more at equal quality,
   and do they compose?
5. Safe boundary for embedding-similarity cache reuse in agentic (stateful) contexts — if any.

**Measurement is built before aggressive techniques** so every claim is data-backed.

---

## 8. Security & privacy posture (summary)

- Trust boundaries threat-modeled in `docs/security/threat-model.md` (STRIDE).
- Proxy holds **provider API keys** → from env/secret store, never logged, never cached
  (`workflow-secrets`). The proxy is a high-value credential holder — least privilege, bound
  to loopback+tailnet, never public (`topic-tailnet-dev-access`).
- Cache/graph data = **confidential**: redact-before-persist, TTL, opt-out patterns, kill
  switch; optional at-rest encryption (M2+).
- Provider responses + any retrieved/cached content treated as **untrusted** downstream
  (`std-owasp-llm` LLM02, `topic-api-consumption`).
- Fail-closed on security ambiguity; fail-open on *availability* (forward original request).
