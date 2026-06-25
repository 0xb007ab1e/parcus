# Threat Model — parsimony proxy

STRIDE over the proxy's data-flow. Revisit when trust boundaries change (e.g. enabling the
hosted/shared mode). Method per `@rules/workflow-threat-model.md`.

## 1. System & data-flow

```
[Harness on host] --(1)--> [parsimony proxy] --(2)--> [Provider API (Anthropic/OpenAI)]
                                  |
                                  +--(3)--> [Local store: SQLite cache + graph]  (at rest)
```

External entities: the harness (local, trusted-ish), the provider (external, untrusted output).
Processes: the proxy. Data stores: the SQLite cache/graph (confidential).

### Trust boundaries
- **TB1 — Harness → proxy (1):** loopback/tailnet HTTP. Carries prompts + **provider API keys**.
- **TB2 — Proxy → provider (2):** outbound TLS. Carries (possibly compressed) prompts + keys;
  returns responses (**untrusted content**).
- **TB3 — Proxy → local store (3):** persistence of **confidential** prompt/response/graph data.

## 2. Data classification
- Prompts/responses: **confidential** (may contain source code, secrets, PII).
- Provider API keys: **restricted** (full account compromise if leaked).
- Token/savings metrics: internal (must be PII-free).

## 3. STRIDE

| # | Threat (STRIDE) | Boundary | Mitigation |
|---|---|---|---|
| S1 | Spoofed client on the bind address (S) | TB1 | Bind **loopback + tailnet only**, never `0.0.0.0`/public (`topic-tailnet-dev-access`); tailnet gated by Tailscale ACLs. Hosted mode adds an optional fail-closed **edge allow-list** of credential-derived tenant ids (401 before forwarding; ADR 0003) atop the provider's own auth. |
| S2 | Proxy forwards to a spoofed/MitM upstream (S) | TB2 | Upstream URLs are an **allow-list** of known provider hosts; TLS verification on; no following redirects to other hosts (`topic-api-consumption`, SSRF). |
| T1 | Tampering with request alters model result (T) | proxy | **Immutable-span** classification (code/paths/quotes/tool-JSON never altered); lossy passes gated by no-regression eval; **fail open** to original request on any doubt. |
| T2 | Tampering with cache store on disk (T) | TB3 | Store in user-owned `0600` dir, `.gitignore`d; **opt-in AES-256-GCM at-rest encryption** of response bodies (AEAD ⇒ tamper/relocation detected via the auth tag, cache key bound as AAD; key from env/keyfile, never VCS; enabling without a valid key fails closed; undecryptable entry ⇒ miss). See ADR 0005. |
| R1 | Repudiation / no audit of what was changed (R) | proxy | Structured logs of transform decisions + cache hit/miss with a correlation id (redacted) (`topic-logging-observability`). |
| I1 | **API key leak** via logs/metrics/cache (I) | TB1/TB3 | Keys **never logged, never cached, never persisted**; forwarded verbatim or injected from env/secret store only; redaction allow-list at the logging boundary (master §5, `workflow-secrets`). **Critical** severity. |
| I2 | PII/secret in persisted cache or graph (I) | TB3 | **Redact-before-persist** (default on); TTL expiry; no-cache regex patterns; kill switch; in-memory-only mode option. |
| I3 | Verbose errors leak internals to the harness (I) | TB1 | Safe error envelope; detail stays server-side (`topic-error-handling`). |
| I4 | Untrusted provider/cached content treated as instructions (I) | TB2 | Responses/cached/retrieved content are **untrusted** — never executed/evaluated; passed through unmodified (`std-owasp-llm` LLM02, `topic-api-consumption` API10). |
| D1 | Proxy outage stalls every harness turn (D) | proxy | **Fail open**: on proxy/internal error, forward original request; upstream timeouts + circuit-break; bounded local work; streaming passthrough. |
| D2 | Cache/graph store unbounded growth (D) | TB3 | TTL + size caps + eviction (`topic-caching`); store is a perf layer, correct when empty. |
| D3 | Compression cost exceeds the savings (D) | proxy | Local transforms are fast + bounded; latency budget per call; heavy local models opt-in only. |
| D4 | One tenant exhausts cost/capacity, degrading others (D) | proxy | Hosted mode: optional per-tenant **token-bucket rate limit** → 429 + Retry-After before forwarding (noisy-neighbour; OWASP LLM04/API4 unrestricted resource consumption). Fail-closed against abuse. See ADR 0003. |
| E1 | Cross-context cache reuse serves A's data to B (E) | TB3 | Exact/normalized-hash key includes full salient context; in hosted/multi-tenant mode the key is namespaced by a tenant id **derived server-side from the inbound credential** (never a client field — BOLA/API1), so one tenant can never read another's cached response. The opt-in **similarity** cache stays off by default and is **tenant- and model-scoped** (no cross-tenant/cross-model similar-serve) behind a high threshold + a no-false-hit precision gate; the default lexical embedder is flagged unsafe. See ADR 0003 + 0004. |
| E2 | Compromised proxy escalates via held keys (E) | proxy | Least privilege; keys scoped to forwarding; no shell/file/network beyond provider allow-list; minimal dependencies (`std-supplychain`). |

## 4. Abuse cases → security tests (master §4)
- Send a request containing a fake API key / AWS secret → assert it is **never** written to the
  cache DB or logs (redaction).
- Configure a no-cache pattern → assert matching requests bypass cache entirely.
- Force a compressor exception → assert the **original** request is forwarded unchanged.
- Point upstream at a non-allow-listed host → assert refusal (no SSRF).
- Two different contexts hashing-adjacent → assert distinct cache keys (no cross-context hit).

## 5. Residual risk / decisions
- Tailnet is multi-device but single-user for v1 → no per-request auth (accepted; revisit for
  hosted mode, which pulls in `topic-multi-tenancy` + `topic-authn-authz`).
- At-rest encryption **delivered** as an opt-in AES-256-GCM layer over the response cache
  (ADR 0005); off by default since the cache lives in a user-owned, git-ignored `0600` dir on a
  trusted host, on by config for shared/backed-up/regulated deployments.

_Last reviewed: 2026-06-25. Owner: project author._
