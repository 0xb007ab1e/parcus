# ADR 0003 — Hosted/multi-tenant mode: credential-derived tenant, namespaced state

- Status: Accepted (foundation slice — cache isolation)
- Date: 2026-06-24
- Deciders: project author

## Context

`parsimony` is local-first: the default deployment is one user running the proxy on loopback /
their tailnet, forwarding their own credential to the provider. A second deployment shape — a
**hosted** proxy shared by multiple principals — was kept in mind from the start (the engine is
ports & adapters; the cache key already took a salt). Multi-tenancy turns tenant isolation into
a **security boundary**: a cross-tenant data leak (one tenant reading another's cached response)
would be a critical incident (`topic-multi-tenancy`, threat E1).

The single largest leak vector is the **response cache**: identical request bodies from two
tenants would, without scoping, collide on the same cache entry and serve one tenant the
other's answer.

## Decision

1. **Tenant is derived server-side from the inbound credential — never from a client field.**
   `parsimony.tenant.derive_tenant` hashes the request's own auth header (`x-api-key`, else
   `authorization`) with the install salt into a short, opaque digest. Trusting a client-asserted
   tenant id would be Broken Object Level Authorization (OWASP API1 / BOLA): any caller could
   read another tenant's data by claiming their id. The raw credential is never stored or logged
   — only the content-free digest.
2. **Per-tenant cache namespacing.** When `multi_tenant` is on, the tenant digest is folded into
   the cache-key salt (`{salt}|t:{tenant}`), so two tenants with byte-identical requests get
   distinct keys and can never share a cached response. A required negative test asserts no
   cross-tenant hit (`test_different_tenants_never_share_cache`).
3. **Off by default; single-tenant behaviour is byte-for-byte unchanged.** With `multi_tenant`
   off the tenant is the empty string, the salt is untouched, and the cache behaves exactly as
   before (correct for the local single-principal deployment).
4. **`tenant.py` is a critical path** (100%-coverage gate) because it is the isolation boundary.

## Edge authorization (slice 2 — delivered)

On top of the provider's own authentication of the forwarded credential, hosted operators get an
optional **fail-closed allow-list** of permitted tenant ids:

- `parsimony.tenant.is_authorized(tenant, allowed)` — empty allow-list = open; non-empty = only
  listed tenants pass (anonymous/unlisted → **401**, never forwarded). In the critical gate.
- `EngineConfig.allowed_tenants` / `Settings.allowed_tenants` (`PARSIMONY_ALLOWED_TENANTS`,
  comma-separated). A `model_validator` **fails fast** if it is set without `multi_tenant` (it
  keys on the tenant id, which only exists in multi-tenant mode).
- `parsimony tenant-id` reads a credential from `PARSIMONY_TENANT_CREDENTIAL` or stdin — **never
  argv** (shell-history / process-table leak; lang-shell, workflow-secrets) — and prints its
  tenant id so operators can build the allow-list without the proxy seeing raw keys at runtime.

This authorizes *which principals may use this instance* and lets an operator revoke a tenant at
the proxy without rotating the provider key — defence in depth, not a replacement for provider auth.

## Scope — all slices delivered

Hosted mode is delivered across five slices: **cache scoping** (1), **edge authorization** (2),
**per-tenant memory isolation** (3), **per-tenant rate limiting** (4), and **per-tenant metrics
tagging** (5 — below). The first four are isolation/abuse controls; the fifth is attribution.

## Per-tenant memory isolation (slice 3 — delivered)

The graph memory holds prior prompt content for injection/compaction. A *shared* graph in hosted
mode would let one tenant's context be retrieved into another's request — the memory analogue of
E1. A `MemoryProvider` seam resolves the right graph for a tenant **before** any ingest/retrieve:

- `parsimony.memory.SharedMemoryProvider` — one graph for all (single-tenant; today's behaviour).
- `parsimony.memory.PerTenantMemoryProvider` — a fresh graph per tenant id, built lazily and
  cached; ingest/retrieval for one tenant can never surface another's content.
- The engine holds a `MemoryProvider` (wrapping the single memory via `SharedMemoryProvider` when
  none is injected) and calls `for_tenant(tenant)` per request; the composition root builds a
  `PerTenantMemoryProvider` when `multi_tenant` is on. `memory.provider` is in the critical gate;
  a negative test proves tenant B cannot retrieve tenant A's ingested content.

This makes hosted memory injection **safe to enable** — though it remains off by default until an
operator opts in.

## Per-tenant rate limiting (slice 4 — delivered)

To bound cost and stop one tenant degrading others (OWASP LLM04 / API4 unrestricted resource
consumption; noisy-neighbour), each tenant gets an independent **token bucket**:

- `parsimony.quota.RateLimiter` — `capacity` burst tokens refilling at `refill_per_sec`; each
  request consumes one. Empty bucket → the engine returns **429 + `Retry-After`** *before* any
  upstream call. Elapsed time uses a **monotonic** source so an NTP step can't grant/revoke
  tokens; buckets are per-key (tenant id). In the 100%-critical gate.
- `Settings.rate_limit_per_minute` / `rate_limit_burst` (`PARSIMONY_RATE_LIMIT_*`); `0` disables
  (default). A `field_validator` rejects negatives. Keyed by the derived tenant id — one shared
  bucket in single-tenant mode, per-tenant when `multi_tenant` is on.

This control **fails closed** against abuse (shed the request), distinct from the optimization
path, which fails open.

## Per-tenant metrics tagging (slice 5 — delivered)

For billing/support attribution, each savings event carries the **opaque, content-free** tenant
id (`SavingsEvent.tenant`; never the raw credential). The persistent store and the in-memory
aggregate both expose a `by_tenant` rollup (requests + token reduction per tenant) — populated
only for credentialed tenants (the single-tenant ``""`` bucket is excluded), surfaced by
`parsimony stats` and the JSON endpoint. The tenant id is **not** emitted as a response header,
so attribution doesn't reveal which tenant served a request. This is attribution, not an
isolation boundary — metrics were already content-free counts. (Prometheus output stays global:
per-tenant labels are deliberately omitted to avoid metric-cardinality blow-up.)

## Consequences

- Hosted operators get cache isolation, an edge allow-list, isolated per-tenant memory,
  per-tenant rate limiting, and per-tenant attribution — all one-flag opt-ins, with no change to
  local single-tenant users.
- The tenant id is opaque and never exposed as a response header, so enabling multi-tenant mode
  does not leak which tenant served a request.
- Hosted mode is feature-complete for this ADR; future hardening (e.g. per-tenant encryption
  keys / crypto-shredding) can build on the same server-side tenant identity.
