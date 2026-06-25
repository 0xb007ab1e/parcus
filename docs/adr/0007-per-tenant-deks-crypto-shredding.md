# ADR 0007 — Per-tenant DEKs & crypto-shredding

- Status: Accepted
- Date: 2026-06-25
- Deciders: project author

## Context

At-rest cache encryption (ADR 0005) used a single key for the whole cache. In hosted/multi-tenant
mode that leaves two gaps: (1) one key protects every tenant's data, so a key compromise exposes
all of them; (2) there's no fast way to honour a tenant's **right to erasure** (GDPR) — you'd have
to find and delete their rows. Both are addressed by giving each tenant its **own** encryption key
and erasing by destroying the key (**crypto-shredding**), composing the ADR 0003 server-side
tenant identity with the ADR 0005/0006 encryption + rotation.

## Decision

1. **Per-tenant DEKs.** In multi-tenant mode each tenant's data-encryption key is
   `HKDF-SHA256(master_key, info="parsimony-cache-tenant:<tenant>")` — distinct per tenant, derived
   lazily and cached. Compromising one tenant's derived key doesn't expose another's. Single-tenant
   mode keeps the single-key path unchanged (no HKDF) for backward compatibility.
2. **Crypto-shredding by withheld key.** Listing a tenant in `PARSIMONY_CACHE_SHREDDED_TENANTS`
   makes the provider return **no cipher** for it (`for_tenant → None`): its existing ciphertext
   becomes immediately inaccessible (reads miss, writes are skipped) — logical erasure **without
   scanning or deleting rows**; the orphaned bodies age out by TTL. Requires encryption +
   multi-tenant (validated; fail fast otherwise).
3. **Tenant threaded through `CachePort`.** `get`/`put` gained an optional `tenant: str = ""`
   (default preserves all existing behaviour; plain stores ignore it). The engine passes the
   credential-derived tenant; only the encrypting cache uses it (to select the DEK). This was the
   minimal seam — the alternative (encoding the tenant in the cache-key string) was rejected as
   hacky.
4. **Rotation composes.** `previous_master_keys` derive each tenant's *previous* DEKs, so a master
   rotation (ADR 0005-rotation) decrypts pre-rotation per-tenant entries during the overlap.
5. **Same primitives, same guards.** Each per-tenant cipher is a `CacheCipher` (AES-256-GCM, cache
   key as AAD, version byte, CSPRNG nonce); `cache.encryption` stays in the 100%-critical gate.

## Consequences

- Stronger isolation (per-tenant key blast-radius) and a one-flag **right-to-erasure** mechanism
  for hosted operators; single-tenant deployments are unaffected.
- **Honest limitation:** shredding withholds the *current* derivation of the tenant's key. Because
  the DEK is derived from the master (not a stored, destroyed secret), *un*-shredding a tenant
  before their old entries' TTL expires would make those entries readable again. So shredding is
  intended **permanent**; for guaranteed irreversibility, keep the tenant shredded until the cache
  TTL has elapsed (or use a short TTL / flush). A monotonic per-tenant key **epoch** (so
  un-shredding can't resurrect old data) is a noted future refinement — it needs persistent
  per-tenant version state, deliberately out of scope here.
- KMS-managed master keys remain a future follow-up (key is still env/keyfile).
