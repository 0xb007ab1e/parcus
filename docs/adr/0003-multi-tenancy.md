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

## Scope of this slice / what's deferred

This ADR covers the **isolation foundation: cache scoping**. The remaining hosted-mode slices,
deferred to keep changes small and reviewable, are:

- **Proxy authentication** — validate an allow-listed credential at the edge before forwarding
  (today the proxy relays whatever credential it receives; hosted mode should authenticate the
  caller). `topic-authn-authz`.
- **Per-tenant memory + metrics isolation** — the graph memory and metrics store are currently
  shared; before hosted memory/injection is enabled they must be tenant-scoped (a shared graph
  would leak across tenants exactly like the cache would have). Memory injection stays off by
  default until then.
- **Per-tenant quotas / rate limits** — bound cost and prevent noisy-neighbour abuse
  (`topic-multi-tenancy`, `std-owasp-api` API4).

## Consequences

- Hosted operators get cache isolation with a one-flag opt-in and no change to local users.
- The digest is opaque and not exposed in any response header, so enabling multi-tenant mode
  does not leak which tenant served a request.
- Until the deferred slices land, hosted mode should run with memory **off** (the default) and
  rely on the proxy's network controls + the provider's own auth for access control.
