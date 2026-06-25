# Changelog

All notable changes to parsimony are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); versioning is [SemVer](https://semver.org).

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
- **Observability** — per-stage reduction + accuracy, persistent metrics, `parsimony stats`,
  JSON + Prometheus endpoints, health endpoint; per-tenant attribution.
- **Hosted/multi-tenant mode** — server-side tenant isolation (cache, memory), edge
  authorization allow-list, per-tenant token-bucket rate limiting.
- **At-rest cache encryption** — AES-256-GCM (AEAD), graceful key rotation, per-tenant derived
  keys (DEKs) + crypto-shredding.
- **Eval harness** — token-reduction metrics with model-free + judge-based gates
  (`parsimony eval [--filler|--retrieval|--similarity]`).
- **Supply chain** — CI security gates (lint, mypy-strict, bandit, tests with 100%-critical /
  ≥90% coverage, pip-audit, SBOM, gitleaks); tag-triggered signed release (SLSA provenance).

### Security
- Provider API keys are never logged, cached, or persisted; cache/graph data redacted before
  persist; binds loopback/tailnet only (never public). Threat model in `docs/security/`.

[0.1.0]: https://github.com/0xb007ab1e/parsimony/releases/tag/v0.1.0
