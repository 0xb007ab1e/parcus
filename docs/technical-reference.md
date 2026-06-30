# parcus — Technical Reference

The shapes of every piece: architecture, the canonical data model, the ports (interfaces), the
request pipeline, each subsystem, the full configuration reference, the HTTP surface, and the
CLI. Per-symbol API docs are generated from docstrings with `make docs` (pdoc); this document is
the hand-written map.

> Version 0.1.0. Import package: `parcus`. Env prefix: `PARCUS_`. Reserved endpoints:
> `/__parcus__/*`. Response headers: `x-parcus-*`.

---

## 1. Architecture

parcus is **ports & adapters** (hexagonal) with a **functional core / imperative shell**:

- **Functional core** — pure logic over the canonical model: span classification, compression
  passes, cache-key hashing, redaction, invariants, retrieval, tenant derivation, rate-limit math.
  No I/O, deterministic, trivially testable.
- **Imperative shell** — the FastAPI app + the `ProxyEngine`: reads bytes, calls the core, does
  the upstream HTTP, persists to SQLite. Side effects live here.
- **Ports** (`parcus.ports`) are `typing.Protocol` interfaces the core depends on; **adapters**
  (httpx upstream, SQLite stores, tiktoken tokenizer, optional local models) implement them and
  are injected at the **composition root** (`parcus.cli`). The core never imports an adapter.

Two cross-cutting rules:

- **Fail open (optimization path).** Any error or ambiguity → forward the **original** request
  and return the real response. Components return inputs unchanged / `None` rather than raising.
- **Fail closed (security path).** A security control (auth, "is this a secret?", missing
  encryption key) denies/aborts on doubt.

See [ADR 0001](adr/0001-proxy-architecture-and-fail-open.md).

## 2. Package map

```
parcus/
  model.py          Canonical data model (Span, Message, CanonicalRequest, …)
  ports.py          Core Protocols (Tokenizer/Compressor/Clock/Redactor/Cache/Memory)
  spans.py          classify_spans(): split text into mutable prose vs immutable spans
  tokenize.py       default_tokenizer(): tiktoken/heuristic token counter
  invariants.py     Model-free equivalence proofs (lossless / filler) — critical path
  tenant.py         derive_tenant() + is_authorized() — server-side tenant id, edge authz
  quota.py          RateLimiter (per-tenant token bucket)
  cli.py            Composition root + `parcus` CLI (serve/eval/stats/tenant-id)
  config/settings.py  Typed PARCUS_* settings (fail-fast validation)
  proxy/
    app.py          FastAPI ingress; reserved endpoints; streaming passthrough
    engine.py       ProxyEngine — the non-streaming pipeline; EngineConfig
    dialects.py     detect/parse/serialize Anthropic + OpenAI (text-only subset)
    upstream.py     HttpxUpstream adapter (UpstreamRequest/Response)
  compress/
    lossless.py     Tier-0 (whitespace, code-aware)
    filler.py       Tier-1 (DEFAULT_FILLERS / AGGRESSIVE_FILLERS)
    learned.py      Tier-2 (LearnedCompressor + TokenReducer + LLMLinguaReducer)
    chain.py        ChainCompressor (compose passes); null.py; sampling.py (VerifySampler)
  cache/
    key.py          compute_key(): salted hash of the canonical request
    policy.py       CachePolicy (no-cache patterns + credential bypass)
    sqlite_cache.py SqliteCache (0600, TTL); null.py; clock.py (SystemClock)
    similarity.py   SimilarityCache (opt-in near-duplicate serve)
    encryption.py   CacheCipher, EncryptedCache, CipherProvider, TenantCipherProvider
  memory/
    graph.py store.py provider.py embedding.py retrieval.py summary.py
    compaction.py ingest.py terms.py model.py   (graph memory: Tracks A/B/C)
  redact/           Redactor + secret/PII patterns
  obs/
    events.py       SavingsEvent + StageStat
    sinks.py        MetricsSink, NullSink, LoggingSink, AggregateSink, MultiSink
    store.py        SqliteMetricsSink (persistent); report.py; prometheus.py
  eval/             Offline measurement + gates (runner/metrics/equivalence/quality/
                    retrieval/similarity/samples/dataset)
```

## 3. Canonical data model (`parcus.model`)

Every request is parsed into a `CanonicalRequest` on the way in and re-serialized to its original
dialect on the way out. All pure logic operates on this model. All types are frozen dataclasses.

**`Span`** — a contiguous run of message text with a mutability flag.

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | The literal text. |
| `mutable` | `bool` (default `True`) | Whether compression may alter it. Code, paths, URLs, quoted text, tool JSON, the trailing instruction are **immutable**. |

`with_text(text)` → a copy with new text, preserving `mutable`.

**`Message`** — one conversation turn.

| Field | Type | Meaning |
|---|---|---|
| `role` | `Role` | `system` / `user` / `assistant` / `tool`. |
| `spans` | `tuple[Span, ...]` | Ordered spans; `.text` concatenates them. |

**`CanonicalRequest`** — a provider-agnostic inference request.

| Field | Type | Meaning |
|---|---|---|
| `dialect` | `Dialect` | `anthropic` / `openai` / `unknown`. |
| `model` | `str \| None` | Requested model id, if known. |
| `messages` | `tuple[Message, ...]` | Conversation turns. |
| `system` | `str \| None` | System prompt (kept separate, as providers do). |
| `stream` | `bool` | Whether a streaming response was requested. |
| `tools_json` | `str \| None` | Verbatim, immutable tool/function schema JSON. |

`.text` concatenates system + all messages (for measurement only).

**`CompressionStats`** — emitted by every compression pass.

| Field | Type | Meaning |
|---|---|---|
| `step` | `str` | Pass id (`lossless` / `filler` / `learned` / `memory`). |
| `tokens_before` / `tokens_after` | `int` | Token counts around the pass. |
| `spans_touched` | `int` | Mutable spans actually modified. |
| `notes` | `tuple[str, ...]` | Optional audit detail. |
| `ok` | `bool \| None` | Result of the pass's **model-free self-check** (invariant held), or `None` if the pass has no runtime invariant (accuracy comes from the offline gate). Drives the live accuracy metric. |

Properties: `tokens_saved` (clamped ≥0), `ratio` (∈ [0,1]).

**`RedactionReport`** — `total: int`, `categories: tuple[str, ...]`, `.has_secrets`.

**`CachedResponse`** — `status_code: int`, `body: bytes` (stored **verbatim** for byte-for-byte
replay), `content_type: str | None`. The store is confidential.

## 4. Ports (`parcus.ports`)

Structural interfaces the core depends on. Each MUST be deterministic/side-effect-free where
noted, and the I/O ports MUST **fail open**.

| Port | Method(s) | Contract |
|---|---|---|
| `TokenizerPort` | `count(text, model=None) -> int` | Deterministic token count; stable heuristic fallback allowed. |
| `CompressorPort` | `compress(req) -> (CanonicalRequest, tuple[CompressionStats, ...])` | Lossless wrt immutable spans; **fail open** (return input + `()` on error). |
| `ClockPort` | `now() -> float` | Injected time (TTLs, testability). |
| `RedactorPort` | `redact(text) -> (str, RedactionReport)`; `has_secret(text) -> bool` | Mask secrets/PII; only on stored/logged content, never the forwarded request or replayed response. |
| `CachePort` | `get(key, *, tenant="") -> CachedResponse \| None`; `put(key, value, ttl, *, tenant="") -> None` | Exact-match store; **fail open**. `tenant` selects a per-tenant key in the encrypting adapter; plain stores ignore it. |
| `MemoryPort` | `ingest(req) -> None`; `relevant(query, *, limit=5) -> tuple[str, ...]` | Graph-backed read-augmenting memory. |

**Auxiliary protocols** (defined beside their adapters, same role):

| Protocol | Module | Method | Purpose |
|---|---|---|---|
| `EmbedderPort` | `memory.embedding` | `embed(texts) -> list[list[float]]` | Local vectors (HashingEmbedder dep-free; SentenceTransformerEmbedder lazy). |
| `TokenReducer` | `compress.learned` | `reduce(text, *, keep_ratio) -> str` | Tier-2 reduction seam (LLMLinguaReducer = local model). |
| `Summarizer` | `memory.summary` | summarize prior turns → short points | ExtractiveSummarizer (model-free) / LLMSummarizer (offline). |
| `MetricsSink` | `obs.sinks` | `record(event) -> None` | Where `SavingsEvent`s go (Null/Logging/Aggregate/Sqlite/Multi). |
| `GraphStore` | `memory.store` | nodes/edges/vectors | InMemory or SQLite-backed graph. |
| `MemoryProvider` | `memory.provider` | `for_tenant(tenant) -> MemoryPort \| None` | Shared (single) vs per-tenant graphs. |
| `CipherProvider` | `cache.encryption` | `for_tenant(tenant) -> CacheCipher \| None` | Static (single key) vs per-tenant DEKs (None = shredded). |
| `StatsSource` | `proxy.app` | `snapshot() -> dict` | Backs the stats/metrics endpoints. |

## 5. The request pipeline (`ProxyEngine`)

`proxy.app` routes non-streaming requests to `ProxyEngine.handle(method, path, headers, body)`,
which wraps `_handle` and emits one `SavingsEvent`. `_handle`:

1. **Detect dialect** from the path (`dialects.detect`). **Route** to the upstream base URL by
   dialect or auth header; unroutable → `502` (forwarded nothing).
2. **Derive tenant** (`tenant.derive_tenant`) when multi-tenant/allow-list is configured — a
   salted hash of the inbound credential, server-side (never a client field).
3. **Authorize** — if an allow-list is set and the tenant isn't on it → `401` (no upstream call).
4. **Rate-limit** — if a limiter is set and the tenant's token bucket is empty → `429` +
   `Retry-After` (no upstream call).
5. **Canonicalize** the body (`dialects.parse`). Un-parseable / unknown → skip to forward (pass
   through unchanged).
6. **Apply memory** (if enabled): ingest into the tenant's graph; optionally compact via
   retrieval (Track B) or rolling summary (Track C). Compacted bodies are **not** cached.
7. **Compress** the (possibly compacted) request through the configured chain; re-serialize to
   the original dialect. Any failure → forward the original bytes.
8. **Cache decision** — if eligible (cacheable, non-stream, no secret, not compacted), compute
   the key (`cache.key.compute_key`, salted, tenant-namespaced) and: exact hit → replay; miss →
   consult the **similarity** cache (if enabled, same model + tenant + threshold) → replay a
   near-duplicate; else continue.
9. **Forward** upstream (`HttpxUpstream`, redirects disabled, dropped hop headers). On a 2xx, with
   a cache key, **store** the response (encrypted if configured) and remember it for similarity.
10. **Return** the result; the wrapper records a `SavingsEvent` (counts only) to the metrics sink.

Outcomes surface as `x-parcus-cache` = `hit` / `miss` / `similar` / `off` / `stream-bypass`.

## 6. Subsystems

**Compression (`compress/`).** Tiers are `CompressorPort`s composed by `ChainCompressor`:
- `LosslessCompressor` (Tier-0): `normalise_whitespace` on mutable spans only; self-checks
  `invariants.is_lossless_equivalent` → `ok`.
- `FillerCompressor` (Tier-1): removes whole tokens whose normalized form is in `DEFAULT_FILLERS`
  (or `AGGRESSIVE_FILLERS`); self-checks `invariants.is_filler_equivalent` (proves *only*
  allow-listed tokens were dropped) → `ok`.
- `LearnedCompressor` (Tier-2): delegates to a `TokenReducer` (`LLMLinguaReducer`, local, lazy).
  Lossy/semantic → `ok=None`; gated **offline** by the answer-preservation judge. Fails open if
  the model/extra is absent.
- `VerifySampler` (`sampling.py`) trims self-check overhead by sampling at
  `PARCUS_INVARIANT_SAMPLE_RATE`.

**Cache (`cache/`).** `compute_key` = salted SHA-256 of the canonical request (prompts never
stored). `CachePolicy` blocks caching for no-cache regex matches and credential-bearing requests.
`SqliteCache` is `0600`, TTL-bound, fail-open. `SimilarityCache` indexes
`(vector, key, model, tenant)` and serves a near-duplicate's response when cosine ≥ threshold
**and** same model **and** same tenant. `EncryptedCache` wraps any `CachePort` and seals bodies
with `CacheCipher` (AES-256-GCM, version byte, CSPRNG nonce, cache key as AAD); a `CipherProvider`
supplies the cipher — `StaticCipherProvider` (one key) or `TenantCipherProvider` (per-tenant
HKDF-derived DEKs; current + previous master keys for rotation; withheld key = crypto-shredding).

**Memory (`memory/`).** A property graph (`graph.py` over a `GraphStore`: in-memory or SQLite +
JSON vectors) with model-free lexical retrieval (Jaccard) and optional local-embedding cosine
(`embedding.py`). `compaction.py` rewrites a request to inject only retrieved context (Track B) or
a rolling summary (Track C), structurally safe (window starts on a user message, context
prepended). `MemoryProvider` gives each tenant an isolated graph. All off by default, behind the
retrieval-recall gate.

**Redaction (`redact/`).** `Redactor` masks secrets/PII (API keys, tokens, PEM, etc.) before
persistence; `has_secret` drives the cache no-cache bypass. Never applied to the forwarded
request or a replayed response.

**Observability (`obs/`).** `SavingsEvent` (content-free per-request counts + `tuple[StageStat]`)
goes to a `MetricsSink`: `LoggingSink` (structured JSON), `AggregateSink` (in-memory),
`SqliteMetricsSink` (persistent, 0600, with per-tenant attribution), `MultiSink`, `NullSink`.
`report.render_stats` (CLI), `prometheus.render_prometheus` (exporter).

**Tenant & quota.** `tenant.derive_tenant` (server-side, salted credential digest),
`tenant.is_authorized` (fail-closed allow-list), `quota.RateLimiter` (per-tenant token bucket,
monotonic clock).

**Eval (`eval/`).** `evaluate` (token-reduction + equivalence over a dataset), `equivalence`
(re-exports `invariants`), `quality` (`KeywordRecallJudge` deterministic / `LLMJudge` offline),
`retrieval` (recall gate), `similarity` (precision/no-false-hit gate), `samples`/`dataset`.

## 7. Configuration

All settings are `PARCUS_<UPPER_SNAKE>` environment variables (or a git-ignored `.env`), parsed
and validated at startup (fail fast). Defaults shown.

| Setting | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind host. Refuses `0.0.0.0`/`::`/empty (loopback/tailnet only). |
| `PORT` | `8787` | Bind port. |
| `TAILNET_IP` | — | Optional tailnet IP for documentation/binding. |
| `ANTHROPIC_UPSTREAM` | `https://api.anthropic.com` | Anthropic base URL. |
| `OPENAI_UPSTREAM` | `https://api.openai.com` | OpenAI base URL. |
| `LOSSLESS` | `true` | Tier-0 lossless pass. |
| `FILLER` | `false` | Tier-1 filler removal (opt-in). |
| `FILLER_AGGRESSIVE` | `false` | Use the larger `AGGRESSIVE_FILLERS` set. |
| `LEARNED` | `false` | Tier-2 local learned compressor (needs `learned` extra). |
| `LEARNED_RATIO` | `0.5` | Target fraction of prose tokens to keep ∈ (0,1]. |
| `CACHE` | `true` | Exact response cache. |
| `CACHE_TTL_SECONDS` | `86400` | Cached-entry TTL. |
| `CACHE_PATH` | `.parcus/cache.sqlite` | Cache DB path (`:memory:` for ephemeral). |
| `CACHE_NOCACHE_PATTERNS` | — | Comma-separated regexes; matches are never cached. |
| `SALT` | — | Per-install cache-key salt (domain separation). |
| `SIMILARITY_CACHE` | `false` | Opt-in near-duplicate cache. |
| `SIMILARITY_THRESHOLD` | `0.97` | Cosine threshold ∈ [0,1] (high = near-duplicate). |
| `SIMILARITY_MAX_ENTRIES` | `2048` | Index cap (FIFO eviction). |
| `SIMILARITY_EMBEDDER` | `local` | `local` (safe; needs `embeddings` extra) or `hashing`. |
| `SIMILARITY_ALLOW_LEXICAL` | `false` | Required to use the unsafe lexical embedder. |
| `CACHE_ENCRYPTION` | `false` | AES-256-GCM at-rest cache encryption. |
| `CACHE_ENCRYPTION_KEY` | — | base64(32 bytes); SecretStr (masked). |
| `CACHE_ENCRYPTION_KEYFILE` | — | Path to a base64 key file (preferred over inline). |
| `CACHE_ENCRYPTION_PREVIOUS_KEYS` | — | Comma-separated retired keys (decrypt-only; rotation). |
| `CACHE_SHREDDED_TENANTS` | — | Tenant ids whose key is withheld (erasure). Needs multi-tenant + encryption. |
| `REDACT` | `true` | Redact secrets/PII before persisting. |
| `LOG_LEVEL` | `INFO` | Log level. |
| `METRICS` | `true` | Emit per-turn savings metrics. |
| `METRICS_PATH` | `.parcus/metrics.sqlite` | Metrics DB path. |
| `INVARIANT_SAMPLE_RATE` | `1.0` | Fraction of requests to run the per-stage self-check on. |
| `MEMORY` | `false` | Ingest requests into the memory graph. |
| `MEMORY_INJECT` | `false` | Track B: compact via retrieval (needs `MEMORY`). |
| `MEMORY_SUMMARIZE` | `false` | Track C: rolling-summary compaction (needs `MEMORY`). |
| `MEMORY_KEEP_RECENT` | `4` | Recent messages kept verbatim when compacting. |
| `MEMORY_RETRIEVE` | `3` | Max retrieved snippets injected. |
| `MEMORY_SUMMARY_ITEMS` | `5` | Max summary points. |
| `MEMORY_MIN_MESSAGES` | `8` | Only compact requests longer than this. |
| `MULTI_TENANT` | `false` | Hosted mode: per-tenant cache/memory isolation. |
| `ALLOWED_TENANTS` | — | Edge authorization allow-list (tenant ids). Needs `MULTI_TENANT`. |
| `RATE_LIMIT_PER_MINUTE` | `0` | Per-tenant rate limit (0 = off). |
| `RATE_LIMIT_BURST` | `0` | Bucket capacity (0 = one minute's worth). |

## 8. HTTP surface

- **Catch-all reverse proxy.** Any path is routed to the provider; non-streaming → engine,
  streaming (`"stream": true`) → `prepare_stream` (authorize + rate-limit + **compress the request
  body**), then forward and relay the SSE response untouched (not response-cached).
- **Reserved local endpoints** (answered by parcus, never forwarded): `GET /__parcus__/health`,
  `GET /__parcus__/stats` (JSON snapshot), `GET /__parcus__/metrics` (Prometheus).
- **Response meta headers**: `x-parcus-cache` (hit/miss/similar/off/stream-bypass),
  `x-parcus-dialect`, `x-parcus-memory`, `x-parcus-tokens-before/after/saved` (parcus's local
  estimate of the *request*), and — on a forwarded non-streaming response — the provider's own
  billed counts `x-parcus-upstream-input/output-tokens` plus its prompt-cache signal
  `x-parcus-upstream-cache-read/write-tokens` (ground truth; `cache-read > 0` means the re-sent
  prefix hit the provider's prompt cache — watch it to confirm compression didn't bust it). The
  tenant id is **never** exposed in a header.
- **Dialects** (`dialects.py`): Anthropic Messages (`/v1/messages`) and OpenAI Chat Completions
  (`/v1/chat/completions`), text-only subset; richer payloads pass through untouched.

## 9. CLI (`parcus …`)

| Command | Purpose |
|---|---|
| `parcus serve [--host H] [--port P]` | Run the proxy (refuses public binds). |
| `parcus eval [dataset] [--filler] [--aggressive] [--retrieval] [--similarity] [--threshold T] [--embedder lexical\|hashing\|local] [--record]` | Measure token reduction + run the matching gate. |
| `parcus stats` | Render aggregated per-stage reduction + accuracy + eval gates. |
| `parcus tenant-id` | Print the tenant id for a credential (from `PARCUS_TENANT_CREDENTIAL` or stdin — never argv) to build the allow-list. |
| `parcus --version` | Print version. |

## 10. Correctness model

- **Fail-open matrix.** Unroutable → 502; unparseable/unknown dialect → pass through; compressor
  error → original bytes; tokenizer error → token metrics drop to 0 (request unaffected);
  memory/similarity/cache error → behave as miss/no-op; cache store error → no-op. Security:
  redactor error → request still forwarded but **not cached** (fails closed for confidentiality);
  missing-tenant on a configured allow-list → 401; over rate limit → 429; encryption enabled
  without a key → refuse to start.
- **Defense in depth at every seam.** The engine does not merely *trust* the port contracts —
  it guards each trusted-adapter call (tokenizer, redactor, cache get/put, similarity
  lookup/remember) so that even a *contract-violating* adapter that raises degrades to
  "skip the optimization, forward the request" rather than crashing. This is the prime
  directive made robust, not just documented.
- **Invariants** (`invariants.py`) are the model-free proofs that make Tier-0/Tier-1 safe without
  a model; Tier-2 and the similarity cache rely on offline quality/precision gates instead.
- **Test layers.**
  - *Example-based* unit/integration tests (`tests/unit`, `tests/integration`).
  - *Property-based* invariant tests (`tests/property`, Hypothesis): for every synthesised
    request, each tier never expands tokens, preserves immutable spans byte-for-byte and request
    structure, satisfies its model-free invariant (lossless = whitespace-only; filler = only
    allow-listed tokens), and is deterministic + idempotent.
  - *Fault-injection* fail-open tests (`tests/integration/test_fail_open.py`): an adapter that
    raises at each seam is asserted to still yield the real upstream response — encoding the
    fail-open matrix above as executable regression tests.
- **Critical-path 100% coverage gate** (CI): `compress`, `model`, `spans`, `cache.key`,
  `cache.policy`, `cache.similarity`, `cache.encryption`, `cache.epoch`, `redact`, `invariants`,
  `eval.equivalence`, `eval.quality`, `memory.compaction`, `memory.provider`, `tenant`, `quota`.
  Repo-wide gate ≥90% line+branch.

---

*See the [guide](guide.md) for the plain-language version, the [FAQ](faq.md) for specific
questions, and the [ADRs](adr/) for why each design choice was made.*
