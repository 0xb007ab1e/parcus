# ADR 0002 — Graph memory: model-free first, eval-gated injection

- Status: Accepted
- Date: 2026-06-24
- Deciders: project author

## Context

`parsimony` cuts tokens two ways: compress the outbound request (M1/M2) and *avoid resending
context*. The latter is the graph-memory work (PLAN §5), with three tracks:

- **A — Response cache** (shipped in M1): skip a call entirely on an exact/normalized-hash hit.
- **B — Context-retrieval graph (RAG-lite):** keep a graph of durable facts/decisions/entities;
  inject only the *relevant subgraph* instead of re-sending big context each turn.
- **C — Conversation compaction:** replace verbose history with a compact derived summary that
  preserves referenced facts.

Tracks B and C *modify what the model sees*, so they carry real correctness risk and must be
measured before they touch the live request path.

## Decision

1. **One `MemoryPort`** (`ingest(request)`, `relevant(query, limit)`) as the seam; concrete
   implementations swap freely (in-memory now, SQLite + sqlite-vec later) without touching the
   core — consistent with the M1 ports & adapters architecture.
2. **Model-free first.** The foundation uses a dependency-free in-memory property graph with
   **lexical** term extraction + Jaccard retrieval. No model, no network. Optional **local**
   embeddings (sentence-transformers, lazy) are an opt-in later slice — never an outbound call
   (a memory that phoned home would defeat the project's purpose).
3. **Injection is deferred and eval-gated.** This slice builds and tests the graph + retrieval
   but does **not** wire memory into the engine. Replacing/augmenting prompt context is only
   turned on once an eval shows it preserves task quality (it cannot use the lossless invariant —
   it changes content by design — so it needs a quality judge / task-success eval, like the
   future Tier-2 learned compressor).
4. **Confidential, like the cache.** Persisted memory is confidential (redact-before-persist,
   TTL, opt-out, kill switch) when the SQLite-backed store lands (master §5).

## Consequences

- (+) A tested retrieval substrate with zero new runtime dependencies; safe to merge because it
  changes no live behavior.
- (+) Clean path to embeddings/persistence/compaction behind the same `MemoryPort`.
- (−) Lexical retrieval is weaker than semantic; acceptable for the foundation and as a
  fast/default tier. Measured against embeddings in a later slice.
- (−) The hard part — proving injected/compacted context preserves quality — is still ahead;
  this ADR commits to gating it on an eval rather than shipping it on faith.

## Alternatives considered

- **Wire retrieval into the proxy now** — rejected: unmeasured context surgery risks degrading
  results, violating "correctness is the gate."
- **Start with a graph database (Neo4j) or embeddings immediately** — rejected for the
  foundation: adds dependencies/services before the retrieval logic is even validated. Kept
  behind the port for later.
