# ADR 0009 — Monotonic per-tenant key epoch (irreversible crypto-shredding)

- Status: Accepted
- Date: 2026-06-30
- Deciders: project author

## Context

Crypto-shredding in ADR 0007 withholds a tenant's derived key via a `shredded` **set**. ADR 0007
itself flagged the gap: the set is in-memory config, so **removing a tenant from it — or simply
restarting without it — makes that tenant's still-unexpired ciphertext readable again**. A true
"right to erasure" needs a shred that cannot be undone.

## Decision

Introduce a per-tenant **key epoch**: an integer folded into the DEK derivation, backed by an
`EpochStore` (port) with an in-memory and a **persistent SQLite** implementation
(`SqliteEpochStore`, `0600`). `EpochCipherProvider` derives
`DEK = HKDF-SHA256(master, info="…:<tenant>:e<epoch>")` at the tenant's **current** epoch.

To shred a tenant, `bump` its epoch. The store is **monotonic** (only `+1`, never down) and
persistent, so:

- the provider then derives a *new* DEK and **never again derives the old one** (it only ever
  reads the current epoch), so pre-bump ciphertext is inaccessible through it and ages out by TTL;
- there is **no un-shred path** — unlike clearing a set, you cannot lower an epoch — and it
  **survives restart**. That closes the ADR 0007 caveat.

Rotation composes: `previous_master_keys` derive the previous DEKs at the *current* epoch.

## Consequences

- Shredding is **operationally irreversible**: no config toggle or restart resurrects the data,
  and the provider will not re-derive a retired epoch's key.
- It is a new opt-in provider; existing `TenantCipherProvider` / `KmsCipherProvider` deployments
  are unaffected (the epoch changes the HKDF `info`, so the two providers' keys never collide and
  must not be mixed on one cache).
- The SQLite epoch store is small and security-relevant; it is added to the **100%-critical
  coverage** gate alongside `cache.encryption`.

## Honest limitation

This is crypto-shredding by **key non-derivation**, not destruction of key material. The master
key is unchanged, so anyone who *separately* holds the master **and** records a tenant's old epoch
number could re-derive the retired DEK and open old ciphertext that has not yet expired. What the
epoch removes is the **application's** ability to do so and any un-shred path — the practical
GDPR-erasure gap from ADR 0007. For hardware-grade erasure, combine this with the KMS master
(ADR 0008): wrap a distinct master per epoch so the retired key truly leaves with the KMS.

## Alternatives considered

- **Keep the withheld-key set, persist it:** persists the *current* shred state but still allows
  un-shred (remove from the set); not monotonic. Rejected — doesn't remove the undo path.
- **Re-encrypt/delete a tenant's rows on shred:** immediate and thorough, but requires scanning
  the store and a writable path; the epoch bump is O(1) and lets entries age out. The two can be
  combined where eager deletion is required.
