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

1. **Off by default; high threshold.** `PARSIMONY_SIMILARITY_CACHE=false`; default cosine
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
5. **A precision gate is the pre-flight.** `parsimony eval --similarity` scores a labelled set of
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
(so `parsimony eval --similarity` exits non-zero out of the box), signalling that:

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
