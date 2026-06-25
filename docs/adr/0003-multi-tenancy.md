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

## Scope / what's still deferred

Delivered: **cache scoping** (slice 1), **edge authorization** (slice 2), and **per-tenant memory
isolation** (slice 3 — below). Remaining hosted-mode slices, deferred to keep changes small:

- **Per-tenant metrics tagging** — metrics are content-free counts (no cross-tenant *leak*), so
  this is tenant attribution for billing/support, not an isolation boundary; lower priority.
- **Per-tenant quotas / rate limits** — bound cost and prevent noisy-neighbour abuse
  (`topic-multi-tenancy`, `std-owasp-api` API4).

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

## Consequences

- Hosted operators get cache isolation, an edge allow-list, and isolated per-tenant memory with
  one-flag opt-ins, and no change to local single-tenant users.
- The digest is opaque and not exposed in any response header, so enabling multi-tenant mode
  does not leak which tenant served a request.
- Metrics remain a shared, content-free aggregate (no leak); per-tenant attribution and quotas
  are the remaining hosted-mode work.
