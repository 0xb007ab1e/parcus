# Changelog

All notable changes to parcus are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); versioning is [SemVer](https://semver.org).

## [Unreleased]

### Added
- **Answer-preservation gate for the lossy tiers (`parcus eval --learned` / `--judged`).** A new
  `eval.judged` harness + `BUILTIN_JUDGED_SAMPLES` judge whether a compressed prompt still
  preserves the required content. `--learned` gates the Tier-2 learned compressor (skips CI-safe
  when the local model is absent; its gate logic is covered in CI via a fake reducer); `--judged
  [--filler --aggressive]` validates a filler set the same way, model-free. Delivers the offline
  gate ADR 0006 deferred.
- **Monotonic per-tenant key epoch (irreversible crypto-shredding).** An `EpochStore` (in-memory +
  persistent SQLite) + `EpochCipherProvider` fold a per-tenant epoch into the DEK derivation; a
  shred is a `bump` that only ever increases and persists, so it has no un-shred path and survives
  restart — closing the ADR 0007 caveat. See ADR 0009.
- **KMS-backed master key for at-rest encryption (envelope encryption).** A `KeyManagementService`
  port + `KmsCipherProvider` let the cache's master key be stored only in KMS-*wrapped* form and
  unwrapped on demand by an injected KMS/HSM adapter (the root key never leaves the KMS). Composes
  with per-tenant DEKs, crypto-shredding, and rotation; no cloud SDK is a dependency (the adapter
  is operator-supplied). See ADR 0008.
- **Property-based invariant tests** (`tests/property/`, Hypothesis): the compression invariants
  (never-expands, immutable spans byte-for-byte, structure preserved, lossless = whitespace-only,
  filler = only allow-listed tokens, deterministic + idempotent) are now checked against
  thousands of synthesised requests, not just hand-written cases.
- **Fault-injection fail-open tests** (`tests/integration/test_fail_open.py`): an adapter that
  raises at each seam (tokenizer, redactor, cache get/put, similarity lookup/remember, memory
  ingest, compressor) is asserted to still serve the genuine upstream response.

### Changed
- **Fail-open hardened (defense in depth).** The engine now guards every trusted-adapter seam,
  so a *contract-violating* adapter that raises degrades to "skip the optimization, forward the
  request" instead of crashing it. A redactor error fails **closed** for caching (the request is
  forwarded but not cached); a tokenizer error drops token metrics to 0 without affecting the
  request.

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

[0.1.0]: https://github.com/0xb007ab1e/parcus/releases/tag/v0.1.0
