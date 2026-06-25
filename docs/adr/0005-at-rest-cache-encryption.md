# ADR 0005 — At-rest cache encryption (opt-in AES-256-GCM)

- Status: Accepted
- Date: 2026-06-25
- Deciders: project author

## Context

The response cache is **confidential** — bodies may contain source code, PII, or secrets
(threat model T2). The backing SQLite file is already `0600`, which suffices for the default
single-user, trusted-host deployment. But on a shared host, a backed-up volume, or for regulated
data (master §5: AES-256 at rest), file permissions alone are not enough. The earlier threat
model deferred at-rest encryption ("M2+"); this ADR delivers it.

## Decision

1. **Opt-in, off by default.** `PARSIMONY_CACHE_ENCRYPTION=false` by default — the local
   single-user case keeps the plain (still `0600`) cache. Operators turn it on for
   shared/backed-up/regulated deployments.
2. **Vetted AEAD, not home-rolled.** `EncryptedCache` decorates any `CachePort` and seals each
   response **body** with **AES-256-GCM** via PyCA `cryptography`'s `AESGCM` (master §5,
   `topic-cryptography` — "never roll your own crypto"). It is an optional `encryption` extra,
   imported lazily, so core parsimony has no new hard dependency.
3. **AEAD with context binding + agility.** Each stored blob is `version || nonce || ct+tag`:
   - a **version byte** for crypto-agility (upgrade the scheme without misreading old entries);
   - a **12-byte CSPRNG nonce** per write (`os.urandom`);
   - the **cache key as additional authenticated data**, binding a ciphertext to its key so it
     can't be relocated and so tampering fails the auth tag.
   The cache key itself is already a one-way hash (`compute_key`), so it is not encrypted; status
   code and content type stay plaintext (not sensitive).
4. **Key handling (`workflow-secrets`).** The key is **base64(32 bytes)** for AES-256, supplied
   via `PARSIMONY_CACHE_ENCRYPTION_KEY` (a pydantic `SecretStr`, masked in reprs/logs) or a
   keyfile — **never** in code/VCS. Enabling encryption **without a valid 32-byte key fails
   closed at startup** (settings validation). Resolution returns `None` on any
   decode/length/read error so the caller fails closed.
5. **Fail open for availability, closed for security.** A blob that can't be decrypted
   (wrong/rotated key, tamper, corruption, legacy plaintext) is treated as a **cache miss**
   (forward upstream) — the cache is a performance layer and must be correct when empty. A
   *security* failure (no key while enabled) fails closed instead. `put` never raises into the
   request path. `cache.encryption` is in the 100%-critical coverage gate.

## Consequences

- Regulated/shared deployments get AES-256 at-rest protection of cached responses with a
  one-flag opt-in and a generated key; the default local experience is unchanged.
- AEAD gives integrity + tamper detection for free (addresses T2 tampering, not just
  confidentiality).
- **Graceful key rotation (delivered).** `CacheCipher` seals with the **current** key and opens
  with the current key plus any configured **previous** keys (`PARSIMONY_CACHE_ENCRYPTION_PREVIOUS_KEYS`,
  decrypt-only). Rotate by promoting a new current key and moving the old one to the previous
  list; entries sealed before the rotation stay readable through the overlap, then age out by
  TTL, after which the old key can be dropped — no cache loss, no plaintext exposure. Trying a
  few keys on a read is cheap and the blob carries no key id (doesn't reveal which key sealed an
  entry). **Crypto-shredding** per tenant (drop a tenant's key to erase their cached data)
  composes naturally with the ADR 0003 tenant identity — a future slice.
- No remote KMS integration yet (keys are env/keyfile); a KMS-backed key provider is a possible
  follow-up for cloud deployments.
