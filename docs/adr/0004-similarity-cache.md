# ADR 0004 — Semantic (similarity) cache: opt-in, gated, embedder-dependent

- Status: Accepted
- Date: 2026-06-25
- Deciders: project author

## Context

The exact response cache (Track A) only hits on a byte-identical canonical request. The PLAN's
M4 names an **embedding-similarity cache mode (opt-in)**: on an exact miss, serve a cached
response for a *near-duplicate* request. Skipping the upstream call is the single largest token
win — but reusing an answer for a *different* prompt **trades correctness for tokens**, so it
needs strong guards (master "correctness is the gate; tokens are the objective").

## Decision

1. **Off by default; high threshold.** `PARCUS_SIMILARITY_CACHE=false`; default cosine
   threshold `0.97` (near-duplicate only). It widens, never replaces, the exact cache — exact
   hits are checked first; similarity is consulted only on an exact miss.
2. **Index keys, not bodies.** `SimilarityCache` stores `(vector, exact-key, model, tenant)`
   only — never prompt content — and fetches the response from the exact cache by key, so TTL,
   eviction, and confidential storage all stay in the exact cache. The index is in-memory and
   FIFO-bounded (a perf layer; correct when empty). *(Persistence across restarts is a possible
   follow-up.)*
3. **Hot-path guards (model-free):** a candidate is served only when **cosine ≥ threshold AND
   same model AND same tenant**. The model guard avoids serving one model's answer for another;
   the **tenant guard is threat E1 again** — a cross-tenant similar-serve would leak data exactly
   like a cross-tenant exact hit. Fails open: any error → forward upstream.
4. **Local embedder; safe by default.** The default is the **local** sentence-transformer
   (`similarity_embedder=local`) — the semantically-safe choice. The dependency-free lexical
   `HashingEmbedder` is available but, being unsafe (see the finding below), requires an explicit
   `similarity_allow_lexical=true` acknowledgement; otherwise enabling the cache with it **fails
   closed at startup**. There is no remote embedder — a cache that called a remote embedding API
   would defeat the project's purpose. *(Update: the original slice defaulted to `hashing`; the
   default was flipped to `local` once the lexical-embedder limitation below was confirmed.)*
5. **A precision gate is the pre-flight.** `parcus eval --similarity` scores a labelled set of
   (anchor, variant, should-hit) pairs and **fails on any false hit** (precision < 1.0). A missed
   paraphrase is lost savings; a served non-paraphrase is a correctness bug — so the gate
   optimises for zero false hits. Operators must pass it with their chosen embedder/threshold
   before enabling. `eval.similarity` and `cache.similarity` are in the 100%-critical gate.

## Key finding — the lexical embedder is unsafe for caching

Running the gate surfaced a real limitation: the dependency-free `HashingEmbedder` is **lexical**
(bag-of-words over `extract_terms`, which drops numbers/short tokens). It therefore **cannot
distinguish requests that differ only in numbers or entities** — e.g. "scale the service to 10
replicas" vs "…2 replicas" embed identically and score cosine **1.0**, a false hit at any
threshold. The built-in adversarial set fails the gate with the lexical embedder **by design**
(so `parcus eval --similarity` exits non-zero out of the box), signalling that:

- **Lexical similarity is not safe for response caching.** The safe choice is the **local
  sentence-transformer** embedder, validated via the gate.
- The dep-free embedder remains useful to exercise/test the mechanism, and as a deliberately
  loud "don't enable this blindly" signal.

## Consequences

- Real near-duplicate token savings are available, opt-in, without changing exact-cache or
  single-tenant behaviour.
- The correctness risk is contained by: off-by-default + high threshold + model/tenant guards +
  a mandatory precision gate + an explicit warning that the default lexical embedder is unsafe.
- Tenant isolation extends to similarity (no cross-tenant similar-serve), consistent with ADR
  0003. Future work: persistent index, and per-tenant similarity tuning.

## Amendment (2026-07-03) — persistent index (snapshot-in-memory)

The index was in-memory only, so it was **cold after every restart** (no near-duplicate hits until
re-warmed). This amendment adds **opt-in persistence** without changing the hot path.

- **Snapshot-in-memory model.** With `similarity_persist=true` the index **hydrates from a durable
  snapshot at startup** and **write-throughs** each `remember`, but every `lookup` still runs
  entirely in memory. Disk I/O happens only at startup and on write — never inside `lookup` — so
  the hot path is unchanged and the cache is still correct when the snapshot is empty/unavailable.
- **Confidential sidecar, posture mirrors the exact cache.** `SqliteSimilarityStore` is a separate
  SQLite file at `similarity_path`, created **`0600`**, storing `(vector, exact-key, model,
  tenant, created_at)` — **still never prompt text**. It persists **plaintext at `0600` when
  `cache_encryption` is off**, exactly as `SqliteCache` persists response bodies. When
  `cache_encryption` is **on**, the vector blob is **sealed with the same `CipherProvider`** as the
  exact cache (built once by `_build_cipher_provider`): AAD-bound to the entry's exact-cache key,
  per-tenant DEKs, and **crypto-shred parity** — a shredded tenant's rows fail to open and are
  **skipped on load** (never served), and are not persisted on write. We deliberately do **not**
  hold vectors to a stricter rule than the response bodies they point to.
- **Classification.** An embedding is prompt-derived **confidential** data (embedding inversion is
  a real research area); on regulated hosts, enable `cache_encryption` and the vectors are sealed
  at rest. The index file joins the exact cache as a `confidential` at-rest store in the threat
  model. Load decodes each row independently, so one shredded/rotated/malformed row is skipped
  rather than sinking the whole snapshot.
- **Fail open.** Any store error (load/append) degrades the index to **in-memory-only** — a broken
  or missing snapshot never breaks the request path.
- **Self-healing staleness.** The snapshot may reference exact-cache keys whose responses have
  since expired/evicted; that just yields a **miss on fetch** (the response still comes from the
  exact cache by key), so a stale index entry is harmless. FIFO cap parity bounds the file.
- Unchanged invariants: same-model + same-tenant guards, high threshold, mandatory precision gate,
  local-embedder default. `cache.*` remains in the 100%-critical gate.
