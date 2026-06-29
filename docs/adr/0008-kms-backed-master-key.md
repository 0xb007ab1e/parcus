# ADR 0008 — KMS-backed master key (envelope encryption)

- Status: Accepted
- Date: 2026-06-29
- Deciders: project author

## Context

At-rest cache encryption (ADR 0005) and per-tenant DEKs (ADR 0007) derive every key from a 32-byte
**master key** sourced from env/keyfile at the composition root. For regulated or shared-host
deployments that is often insufficient: the master sits in plaintext in process env, and there is
no central control over its lifecycle. The standard remedy is **envelope encryption** — keep the
root key in a KMS/HSM that never releases it, store only a *wrapped* (KMS-encrypted) master, and
have the KMS unwrap it on demand.

parcus is local-first and deliberately carries **no cloud SDK dependency**, so it cannot hard-wire
AWS/GCP/Vault clients.

## Decision

Add a `KeyManagementService` **port** (`decrypt_key(wrapped) -> bytes`) and a `KmsCipherProvider`
that implements the existing `CipherProvider` seam:

- Config stores only the **KMS-wrapped** master (ciphertext — safe to commit); the plaintext
  master exists only transiently in memory.
- On first use the provider calls the injected KMS to unwrap the master **once** (result cached),
  then derives per-tenant DEKs via the same HKDF as `TenantCipherProvider` — so KMS sourcing
  **composes** with per-tenant keys, crypto-shredding, and rotation (`previous_wrapped_master_keys`
  unwrap retired masters for the decrypt-only overlap).
- The concrete KMS **adapter** (AWS KMS `Decrypt`, GCP KMS, Vault transit, an HSM) is injected at
  the composition root by the operator; no SDK enters parcus's dependency graph
  (`topic-architecture-patterns`, `workflow-secrets`).

## Consequences

- The root key never leaves the KMS; parcus never persists or logs the plaintext master.
- A KMS failure is **not swallowed** in the provider (a misconfiguration is loud); the engine's
  cache calls fail open around it, so the request is still served — uncached. Net: fails **closed**
  for caching, **open** for availability (`topic-error-handling`, master §5).
- KMS is hit once per process (lazy + cached), not per request — bounded cost/latency.
- Wiring a specific cloud KMS into `parcus serve` is left to deployment (the adapter is operator-
  supplied); the reusable mechanism (port + provider) ships here, fully tested with a fake KMS.

## Alternatives considered

- **Master key direct from KMS into env at boot** (e.g. an init container calls KMS, exports the
  key): simpler, but the plaintext master then lives in env for the process lifetime — exactly
  what envelope encryption avoids. Rejected as the default; still possible via the existing
  `TenantCipherProvider`/`StaticCipherProvider` if an operator prefers it.
- **Bundling a cloud SDK** (boto3/google-cloud-kms): rejected — violates local-first / no-cloud-dep
  and couples the core to one vendor. The injected port keeps it vendor-neutral.
- **Per-entry data keys wrapped by KMS** (a fresh DEK per cached value, KMS-wrapped alongside it):
  maximal isolation but a KMS round-trip per write — too costly for a cache. Per-tenant DEKs from
  one unwrapped master is the right granularity here.
