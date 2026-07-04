# ADR 0010 — Gemini explicit context-cache adapter

- Status: Proposed
- Date: 2026-07-04
- Deciders: project author
- Related: ADR 0001 (ports & adapters, fail-open), ADR 0003 (multi-tenancy),
  ADR 0007 (per-tenant DEKs); `docs/design/token-reduction-frontier.md` §4.1;
  `docs/design/token-reduction-roadmap.md` §2.1 (M1a/M1b cache strategy)

## Context

The frontier survey (`docs/design/token-reduction-frontier.md`) ranks **Gemini explicit
context caching** as the top next feature: it is the one provider caching model parcus does
not yet support, and it is the **safest possible** token win — it changes billing and
transport only, never a single byte of request or response content, so it needs **no
answer-preservation gate** (unlike the Tier-2 learned compressor, ADR 0006).

parcus already models the two caching families it meets today, both via the **pure**
`CacheStrategy` port (`src/parcus/cache/strategy.py`):

- **`EXPLICIT_BREAKPOINT`** (Anthropic) — preserve a stable prefix (M1a) and inject a
  `cache_control` breakpoint (M1b). Both are pure request→request transforms.
- **`AUTOMATIC_PREFIX`** (OpenAI/DeepSeek) and **`NONE`** (Groq) — preserve-only or
  cache-neutral; `NullCacheStrategy` is the fail-open default for any unknown dialect.

Gemini's mechanism is structurally different. It is a **stateful resource lifecycle**, not a
per-request annotation:

1. `cachedContents.create` registers a large stable prefix and returns an opaque **handle**
   (`cachedContent` name) — a **network call** that **incurs a per-hour storage charge**.
2. Later `generateContent` calls **reference the handle** instead of re-sending the prefix
   inline; the cached tokens bill at a discount.
3. The handle has a **TTL** and must be **refreshed or deleted**; stale handles cost money and
   eventually expire server-side.

This collides head-on with the `CacheStrategy` contract, which is documented as **"pure and
deterministic — no I/O, no tokenizer"** (`src/parcus/ports.py`). A context-cache registrar
must do network I/O, hold mutable handle state, and reason about cost and time. Overloading the
pure policy port with it would break the functional-core/imperative-shell split that ADR 0001
rests on. **So the frontier doc's shorthand "extend the `CacheStrategy` port" is refined here:
policy stays pure; the stateful registration lifecycle is a new, separate port in the shell.**

There is also no `GEMINI` dialect yet — `Dialect` is `{ANTHROPIC, OPENAI, UNKNOWN}` — so the
slice necessarily adds Gemini request/response parsing and serialisation
(`generateContent` shape) as a prerequisite.

## Decision

1. **Add the caching taxonomy, keep policy pure.** Introduce
   `CacheModel.EXPLICIT_CONTEXT_API`, a `Dialect.GEMINI`, and a **pure** `GeminiCacheStrategy`
   that implements `capability` + `cacheable_boundary` (which leading messages form the
   stable, must-not-perturb prefix) exactly like the Anthropic strategy. Its `annotate` is a
   **no-op** — there is no in-request marker to inject; referencing a cache is a shell action,
   not a content edit. This preserves the "compress only the volatile tail" M1a guard for
   Gemini for free.

2. **New stateful port for the lifecycle — `ContextCacheRegistrar` (imperative shell).** A
   `typing.Protocol` in `ports.py`, injected like every other I/O port, with roughly:
   - `ensure(prefix, *, model, tenant) -> Handle | None` — return a live handle for this
     (prefix-hash, model, tenant), creating one via the provider if absent and worthwhile;
     return `None` to mean "not cached — send inline" (the fail-open answer).
   - `evict_expired(*, now)` / TTL-driven deletion of stale handles.
   The **only** production adapter is a Gemini one (httpx, async); a fake drives the core in
   tests. The pure engine calls the registrar **through the port**, never the SDK directly.

3. **Handle store is tenant-scoped, local, and not a secret.** Handles persist in the existing
   SQLite store, keyed by **`(tenant, model, prefix_hash)`** so a handle is **never** reused
   across tenants or models (ADR 0003 isolation). A handle is an opaque provider reference, not
   a credential — but it is confidential runtime state, so it lives under the same
   tenant-scoped access rules as cache rows. The provider **API key remains the crown jewel**
   (env/secret store only, never logged, never persisted — master §1, `workflow-secrets`).

4. **Fail open, always — this path never changes content.** On *any* uncertainty — registrar
   error, provider 4xx/5xx/timeout, prefix below the provider minimum, unstable/short-lived
   prefix, handle miss, or the `google` extra absent — the engine forwards the **original
   request with the prefix inline**, uncached. The worst case is "no cache discount this turn,"
   exactly the M1a fail-open posture. Because no content transform occurs, **no answer-
   preservation eval gate is required** (contrast ADR 0006 Tier-2).

5. **Respect the provider minimum; do not register below it.** Explicit context caching has a
   provider-declared minimum prefix size and a minimum useful TTL; below either, registration
   is pure cost with no benefit. The engine (which owns the tokenizer) enforces
   `capability.min_prefix_tokens` before asking the registrar to `ensure`. **The exact Gemini
   minimum is drift-prone — verify against
   [Gemini caching docs](https://ai.google.dev/gemini-api/docs/caching) at implementation time**
   rather than hard-coding a remembered number; encode it as a single named constant.

6. **Cost is bounded and opt-in.** Off by default behind a settings flag (mirroring
   `cache_inject` / `learned`). Because a registered cache **spends** (storage/hour), the
   adapter weighs expected reuse against storage cost, deletes on TTL expiry, and honours an
   operator-set **cap on concurrent cached-content resources / spend**; over the cap → fail
   open (send inline). This keeps the "never make a call / never spend to save tokens without
   opt-in" tenet honest (`topic-token-optimization`, `workflow-gated-actions` spend).

7. **Confidentiality: same content, now retained server-side for the TTL.** The registered
   prefix is byte-identical to what would otherwise be sent inline every turn — no *new* data
   leaves the boundary — but it is now **stored on the provider for the TTL**. Redaction runs
   **before** registration (master §5); document this retention in the threat model and gate it
   behind the opt-in flag for confidential workloads.

8. **Distinct from Gemini implicit caching.** This is the *explicit* `cachedContents` API
   (client-managed handle), not Gemini's automatic implicit prefix caching — which, if/when
   relevant, maps to the existing `AUTOMATIC_PREFIX` model with no registrar.

## Consequences

- (+) The safest token lever in the survey ships: a real discount on Gemini traffic with
  **zero correctness risk** and **no new eval gate** — pure billing/transport.
- (+) The functional-core/imperative-shell boundary (ADR 0001) is strengthened, not eroded:
  caching *policy* stays pure and unit-testable without mocks; caching *I/O and state* live in
  one injected adapter, driven in tests by a fake registrar. The engine reaches 100% on the
  decision path; the Gemini SDK body is `# pragma: no cover` (exercised only with the extra).
- (+) Adds first-class **Gemini dialect** support (parse/serialise `generateContent`), useful
  beyond caching.
- (−) First **stateful, cross-request** feature in the optimization path — introduces handle
  lifecycle, TTL/eviction, and a per-tenant handle store to get right (staleness, races on
  `ensure`, orphaned paid resources). Mitigated by fail-open (a lost/expired handle just means
  inline send) and a spend cap.
- (−) New optional dependency (`google` extra, lazy-imported) and a provider surface to
  maintain as the Gemini API evolves; mitigated by the pass-through/fail-open fallback.
- (−) Confidential prefixes are retained on the provider for the TTL — a documented,
  opt-in-gated data-handling trade-off, not a silent default.

## Alternatives considered

- **Overload the pure `CacheStrategy` port with the registration I/O** — rejected. It would put
  network calls and mutable state behind an interface contractually promised to be pure,
  breaking the core's testability-without-mocks and the ADR 0001 split. Keeping a separate
  stateful port is the whole point.
- **Stateless, per-request re-registration** — rejected. Creating a cached content every turn
  pays the write/storage cost without amortising it; the discount only materialises across
  reuses of a held handle. State is intrinsic to the mechanism.
- **Map Gemini onto `AUTOMATIC_PREFIX` and rely on implicit caching only** — rejected as the
  primary decision (leaves the larger, client-controlled explicit discount on the table); the
  implicit path is cache-neutral and already covered by the null/automatic strategy if desired.
- **Abstractive/extractive request compression for Gemini instead** — deferred. That is the
  §4.2 lever: higher-leverage but content-transforming, so it needs a local model + a new
  answer-preservation gate. This ADR takes the zero-risk win first (frontier §6 sequencing).

## Update (2026-07-04): scaffold landed; the registrar port is async

The seams shipped in PR #84 (`feat/gemini-context-cache-scaffold`): `Dialect.GEMINI`,
`CacheModel.EXPLICIT_CONTEXT_API`, `ContextCacheHandle`, the pure `GeminiCacheStrategy` (registered),
the `ContextCacheRegistrar` port, `NullContextCacheRegistrar` + `GeminiContextCacheRegistrar`
(lifecycle tested via injected create/delete), the opt-in settings, and the optional `gemini` extra.
An independent review confirmed the additions are **inert** on the live path (`detect()` never yields
`GEMINI`; `_inject_cache_breakpoint` gates on `EXPLICIT_BREAKPOINT`) and the lifecycle policy is sound.
It surfaced one decision that is cheapest to make now, while the port has no callers.

**Decision — the registrar port is asynchronous.** `ContextCacheRegistrar.ensure` /
`evict_expired` are `async def`, and the production factory is backed by the async Gemini client
(`client.aio.caches.*`). This refines Decision 2. Rationale:

- The engine forward path is **async** (httpx). The real `genai.Client.caches.create/delete` calls
  **block**; invoking a *sync* registrar that does blocking network I/O on the event loop would stall
  it (`topic-concurrency` / `topic-reliability`: never block the loop with sync I/O).
- Offloading a sync port to a threadpool instead would reintroduce a **data race** on the unguarded
  `_handles` dict. An async port keeps the registrar single-tasked on the loop (atomic between
  `await`s) with no lock, matching how the engine already treats its other collaborators.
- This is what the original Decision 2 intended ("httpx, async"); the scaffold shipped **sync
  placeholders** only because nothing called them yet. Converting the port to async is the **first
  task of the wiring slice**, before any caller depends on the sync shape — so the change is confined
  to the port + registrars + their tests, at no extra cost versus having done it now.

**Also folded into the wiring slice (review findings, non-blocking):**

- **Bound the handle map.** `ensure` currently sheds expired entries only for the requested key or on
  `evict_expired()`; opportunistically prune (or sweep on the miss path) so in-memory growth is
  structural, not dependent on an external scheduler (`topic-resource-management`). Provider *spend*
  is already bounded by the live-handle cap.
- **Redaction before registration is a hard requirement.** The scaffold's `ensure(prefix, …)` cannot
  enforce §7; the engine path MUST redact before calling `ensure`, with an explicit test asserting it.
- **Validate the new settings** (`Field(ge=1)` on `gemini_cache_ttl_seconds` /
  `gemini_cache_max_entries`) and ensure `google-genai` resolves into the lockfile **with hashes**
  when the `gemini` extra is locked (`std-supplychain`).

**Status stays `Proposed`** until the wiring slice (Gemini request routing — detect
`generateContent` → parse → serialise a `cachedContent` reference — plus engine integration and a
live-key validation) lands and flips it to `Accepted`.
