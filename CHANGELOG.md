# Changelog

All notable changes to parcus are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); versioning is [SemVer](https://semver.org).

## [Unreleased]

### Changed
- **Accurate token measurement via `tiktoken` (was a 4-chars/token heuristic).** `default_tokenizer`
  now uses a real BPE encoding when available (`TiktokenTokenizer`, exact for OpenAI-family models,
  heuristic fallback offline). Real-provider validation showed the old heuristic **overstated
  savings ~1.5–2.2×** because it over-counted the whitespace/filler compression removes; a real BPE
  encoding tracks the provider's tokenizer with a near-constant offset, so the reported saved-token
  delta is now accurate. `x-parcus-tokens-*` are parcus's measurement of the request text;
  `x-parcus-upstream-*` remain the provider's billed truth. See `docs/validation/RESULTS.md`.

### Added
- **Real-provider validation evidence** (`docs/validation/RESULTS.md`): parcus in front of Groq
  (OpenAI-compatible), 10 passes × 4 prompt sizes — ground-truth savings (11–21%, ~19% overall,
  Groq's own tokenizer), ~0 ms latency overhead, identical answers, plus live streaming + cache.
- **Streaming requests are now compressed.** A streaming request (`"stream": true`) previously
  bypassed the engine entirely — no compression, and (a latent gap) no authorization or rate
  limiting. `ProxyEngine.prepare_stream` now applies the full request-side pipeline (route →
  server-side tenant + edge authz → rate limit → canonicalize → memory → compress) and forwards
  the **compressed** body, while the SSE **response** still streams back byte-for-byte and
  unbuffered. Streaming responses remain uncached. This is the change that lets parcus actually
  save tokens in front of streaming harnesses like Claude Code; `x-parcus-tokens-*` headers now
  appear on streamed responses too.
- **Provider-usage capture (ground-truth tokens + prompt-cache signal).** parcus now parses the
  provider's `usage` from forwarded non-streaming responses (Anthropic + OpenAI) into a
  `ProviderUsage` and surfaces it on `SavingsEvent.upstream_usage` and `x-parcus-upstream-*`
  response headers — the billed input/output tokens plus `cache-read`/`cache-write` counts. This
  turns "savings" from a local-tokenizer estimate into ground truth and makes the provider
  prompt-cache interaction observable (did request compression preserve the cache hit or bust it
  — PLAN research Q3). Read-only and fail-open; streaming-response usage is captured separately
  with the streaming-request work.

## [0.2.0] - 2026-06-30

Hardening, two opt-in at-rest-encryption features, the lossy-tier correctness gate, and a large
investment in test depth. All changes are backward-compatible and off by default except where
noted; single-user local behaviour is unchanged.

### Added
- **KMS-backed master key for at-rest encryption (envelope encryption).** A `KeyManagementService`
  port + `KmsCipherProvider` let the cache's master key be stored only in KMS-*wrapped* form and
  unwrapped on demand by an injected KMS/HSM adapter (the root key never leaves the KMS). Composes
  with per-tenant DEKs, crypto-shredding, and rotation; no cloud SDK is a dependency (the adapter
  is operator-supplied). See ADR 0008.
- **Monotonic per-tenant key epoch (irreversible crypto-shredding).** An `EpochStore` (in-memory +
  persistent SQLite) + `EpochCipherProvider` fold a per-tenant epoch into the DEK derivation; a
  shred is a `bump` that only ever increases and persists, so it has no un-shred path and survives
  restart — closing the ADR 0007 caveat. See ADR 0009.
- **Answer-preservation gate for the lossy tiers (`parcus eval --learned` / `--judged`).** A new
  `eval.judged` harness + `BUILTIN_JUDGED_SAMPLES` judge whether a compressed prompt still
  preserves the required content. `--learned` gates the Tier-2 learned compressor (skips CI-safe
  when the local model is absent; its gate logic is covered in CI via a fake reducer); `--judged
  [--filler --aggressive]` validates a filler set the same way, model-free. Delivers the offline
  gate ADR 0006 deferred.
- **Expanded test depth.** Property-based compression-invariant tests (Hypothesis); fault-injection
  fail-open tests at every adapter seam; SSE streaming-fidelity (byte-exact / incremental /
  backpressure); concurrency/race tests (rate-limiter over-admission, cache thread-safety,
  cross-tenant under load); credential-safety (the provider key never reaches metrics/headers/cache).
- **Dependency-free QA harnesses** under `qa/` (external/ephemeral tools — nothing added to
  `pyproject`): mutation testing (`make mutate`, mutmut), k6 load/soak, schemathesis HTTP-edge fuzz,
  Atheris parser fuzzing, OWASP ZAP DAST, and a promptfoo real-model lossy-tier eval.

### Changed
- **Fail-open hardened (defense in depth).** The engine now guards every trusted-adapter seam, so a
  *contract-violating* adapter that raises degrades to "skip the optimization, forward the request"
  instead of crashing it. A redactor error fails **closed** for caching (forwarded, not cached); a
  tokenizer error drops token metrics to 0 without affecting the request.
- CI actions bumped from the deprecated Node20 runtime to **Node24**, still pinned by commit SHA.

### Fixed
- SQLite-backed stores close their connection on GC (a finalizer backstop), eliminating the
  `ResourceWarning: unclosed database` noise in the test suite (96 → 0).

## [0.1.0] - 2026-06-25

First tagged release: a local-first, token-thrift inference proxy for agentic harnesses. Every
optimization is **fail-open** and **off by default** except Tier-0 compression and the exact
cache; single-user local behaviour is the default and security controls fail closed.

### Added
- **Transparent proxy** (FastAPI + httpx) speaking Anthropic Messages + OpenAI Chat Completions,
  with pass-through and streaming passthrough; routes by path/credential.
- **Tiered request compression** — Tier-0 lossless (whitespace, code-aware; default on), Tier-1
  filler removal (allow-listed; opt-in, with a model-free guardrail; default + aggressive sets),
  Tier-2 learned (local LLMLingua; opt-in, offline answer-preservation gate). Mutable-span-only.
- **Response cache** — exact/normalized-hash (salted; prompts never stored), plus an opt-in
  **semantic (near-duplicate) cache** (local embedder, tenant+model-scoped, precision-gated;
  safe local embedder by default).
- **Graph memory** — context-retrieval (Track B) and rolling-summary compaction (Track C),
  off by default, behind a retrieval-recall gate; local embedders; SQLite persistence.
- **Observability** — per-stage reduction + accuracy, persistent metrics, `parcus stats`,
  JSON + Prometheus endpoints, health endpoint; per-tenant attribution.
- **Hosted/multi-tenant mode** — server-side tenant isolation (cache, memory), edge
  authorization allow-list, per-tenant token-bucket rate limiting.
- **At-rest cache encryption** — AES-256-GCM (AEAD), graceful key rotation, per-tenant derived
  keys (DEKs) + crypto-shredding.
- **Eval harness** — token-reduction metrics with model-free + judge-based gates
  (`parcus eval [--filler|--retrieval|--similarity]`).
- **Supply chain** — CI security gates (lint, mypy-strict, bandit, tests with 100%-critical /
  ≥90% coverage, pip-audit, SBOM, gitleaks); tag-triggered signed release (SLSA provenance).

### Security
- Provider API keys are never logged, cached, or persisted; cache/graph data redacted before
  persist; binds loopback/tailnet only (never public). Threat model in `docs/security/`.

[0.2.0]: https://github.com/0xb007ab1e/parcus/releases/tag/v0.2.0
[0.1.0]: https://github.com/0xb007ab1e/parcus/releases/tag/v0.1.0
