"""At-rest encryption for the response cache (opt-in defense-in-depth).

The cache is **confidential** — responses may contain source code, PII, or secrets (threat T2).
The backing file is already ``0600``, but on a shared or backed-up host that is not enough for
regulated data. This module encrypts each cached response body with **AES-256-GCM** (an AEAD)
before it is persisted, via an :class:`EncryptedCache` decorator over any :class:`CachePort`.

Design (master §5, ``topic-cryptography``):

* **Vetted primitive, not home-rolled** — PyCA ``cryptography``'s ``AESGCM`` (optional
  ``encryption`` extra; imported only when this module is used).
* **AEAD with context binding** — the cache key is passed as additional authenticated data, so a
  ciphertext is bound to its key and cannot be relocated; tampering fails the auth tag.
* **Unique nonce per write** from a CSPRNG (``os.urandom``).
* **Crypto-agility** — every blob is ``version || nonce || ciphertext+tag``; the version byte
  lets the scheme be upgraded without misreading old entries.
* **Fail open for availability** — a blob that can't be decrypted (wrong/rotated key, tamper,
  corruption) is treated as a **miss** (forward upstream), never an error or garbage served.
  The key itself comes from a secret store/env at the composition root, never from code/VCS
  (``workflow-secrets``); enabling encryption without a valid key fails **closed** at startup.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from parcus.model import CachedResponse
from parcus.ports import CachePort

__all__ = [
    "CacheCipher",
    "CipherProvider",
    "EncryptedCache",
    "StaticCipherProvider",
    "TenantCipherProvider",
]

_VERSION = b"\x01"  # AES-256-GCM, 12-byte nonce. Bump to add/upgrade schemes (crypto-agility).
_NONCE_LEN = 12
_KEY_LEN = 32  # AES-256
_HEADER_LEN = len(_VERSION) + _NONCE_LEN
# HKDF context: namespaces derived per-tenant keys to this purpose (domain separation).
_HKDF_INFO_PREFIX = b"parcus-cache-tenant:"


class CacheCipher:
    """AES-256-GCM sealer/opener for cache values, with graceful key rotation.

    New values are always sealed with the **current** key. Opening tries the current key then
    each **previous** key in turn, so entries written before a rotation stay readable during the
    overlap window (rotate the key → move the old key to ``previous_keys`` → old entries decrypt
    until they expire by TTL, then drop the old key). The blob format carries no key id; trying a
    handful of keys on a read is cheap and avoids leaking which key sealed an entry.

    Args:
        key: The current 32-byte (AES-256) key — used for both sealing and opening.
        previous_keys: Retired keys kept for **decryption only** during a rotation window.

    Raises:
        ValueError: If any key is not exactly 32 bytes.
    """

    def __init__(self, key: bytes, *, previous_keys: tuple[bytes, ...] = ()) -> None:
        """Validate key lengths; build the sealing AEAD and the ordered opening AEADs."""
        for candidate in (key, *previous_keys):
            if len(candidate) != _KEY_LEN:
                raise ValueError(f"cache encryption key must be {_KEY_LEN} bytes (AES-256)")
        self._sealer = AESGCM(key)
        # Current key first (the common case), then retired keys for the rotation overlap.
        self._openers = (self._sealer, *(AESGCM(k) for k in previous_keys))

    def seal(self, aad: str, plaintext: bytes) -> bytes:
        """Return ``version || nonce || ciphertext+tag`` sealed with the **current** key."""
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = self._sealer.encrypt(nonce, plaintext, aad.encode("utf-8"))
        return _VERSION + nonce + ciphertext

    def open(self, aad: str, blob: bytes) -> bytes | None:
        """Return the plaintext, or ``None`` if no key decrypts it (bad version/tamper/wrong key).

        Tries the current key then each previous key, so a value sealed before a rotation still
        opens during the overlap window.
        """
        if len(blob) < _HEADER_LEN or blob[:1] != _VERSION:
            return None
        nonce = blob[len(_VERSION) : _HEADER_LEN]
        ciphertext = blob[_HEADER_LEN:]
        data = aad.encode("utf-8")
        for aead in self._openers:
            try:
                return aead.decrypt(nonce, ciphertext, data)
            except InvalidTag:
                continue
        return None


@runtime_checkable
class CipherProvider(Protocol):
    """Resolves the :class:`CacheCipher` to use for a tenant (``None`` = no caching for it)."""

    def for_tenant(self, tenant: str) -> CacheCipher | None:
        """Return the cipher for ``tenant``, or ``None`` (e.g. shredded → key withheld)."""
        ...


class StaticCipherProvider:
    """One cipher for every tenant — the single-key case (no per-tenant DEKs)."""

    def __init__(self, cipher: CacheCipher) -> None:
        """Hold the single cipher."""
        self._cipher = cipher

    def for_tenant(self, tenant: str) -> CacheCipher | None:
        """Return the single cipher regardless of tenant."""
        return self._cipher


class TenantCipherProvider:
    """Per-tenant **data encryption keys (DEKs)** derived from a master key, with crypto-shredding.

    Each tenant's DEK is ``HKDF-SHA256(master_key, info="…:<tenant>")`` — distinct per tenant, so
    compromising one tenant's derived key doesn't expose others. **Crypto-shredding**: listing a
    tenant in ``shredded`` withholds its key (``for_tenant`` → ``None``), so its cached entries
    become cryptographically inaccessible at once (reads miss, writes skip) without scanning rows
    — the remaining ciphertext bodies just age out by TTL. Rotation composes:
    ``previous_master_keys`` derive each tenant's previous DEKs for decrypt-only during overlap.

    Args:
        master_key: The 32-byte current master key.
        previous_master_keys: Retired master keys (decrypt-only) for rotation overlap.
        shredded: Tenant ids whose key is withheld (erased).
    """

    def __init__(
        self,
        master_key: bytes,
        *,
        previous_master_keys: tuple[bytes, ...] = (),
        shredded: frozenset[str] = frozenset(),
    ) -> None:
        """Validate the master keys and hold them plus the shred-set; DEKs are derived lazily."""
        for candidate in (master_key, *previous_master_keys):
            if len(candidate) != _KEY_LEN:
                raise ValueError(f"cache encryption key must be {_KEY_LEN} bytes (AES-256)")
        self._master = master_key
        self._previous_masters = previous_master_keys
        self._shredded = shredded
        self._cache: dict[str, CacheCipher] = {}

    @staticmethod
    def _derive(master: bytes, tenant: str) -> bytes:
        """Derive a tenant's 32-byte DEK from a master key (HKDF-SHA256, tenant as info)."""
        hkdf = HKDF(
            algorithm=SHA256(),
            length=_KEY_LEN,
            salt=None,
            info=_HKDF_INFO_PREFIX + tenant.encode("utf-8"),
        )
        return hkdf.derive(master)

    def for_tenant(self, tenant: str) -> CacheCipher | None:
        """Return ``tenant``'s cipher (current + previous DEKs), or ``None`` if shredded."""
        if tenant in self._shredded:
            return None
        cipher = self._cache.get(tenant)
        if cipher is None:
            cipher = CacheCipher(
                self._derive(self._master, tenant),
                previous_keys=tuple(self._derive(m, tenant) for m in self._previous_masters),
            )
            self._cache[tenant] = cipher
        return cipher


class EncryptedCache:
    """A :class:`~parcus.ports.CachePort` that encrypts response bodies at rest.

    Wraps an inner cache (e.g. :class:`~parcus.cache.sqlite_cache.SqliteCache`): bodies are
    sealed on ``put`` and opened on ``get`` with the tenant's cipher (resolved per request); status
    code, content type, TTL, and the (already one-way-hashed) key are handled by the inner cache
    unchanged. Fails open. When the provider withholds a tenant's key (shredded), reads miss and
    writes are skipped.

    Args:
        inner: The backing cache that persists the (encrypted) values.
        cipher: A single cipher (single-key mode) — mutually exclusive with ``provider``.
        provider: A :class:`CipherProvider` for per-tenant DEKs — exclusive with ``cipher``.
    """

    def __init__(
        self,
        inner: CachePort,
        cipher: CacheCipher | None = None,
        *,
        provider: CipherProvider | None = None,
    ) -> None:
        """Hold the inner cache and the cipher provider (a lone ``cipher`` is wrapped as static)."""
        if (cipher is None) == (provider is None):
            raise ValueError("EncryptedCache requires exactly one of cipher or provider")
        self._inner = inner
        self._provider: CipherProvider = provider or StaticCipherProvider(cipher)  # type: ignore[arg-type]

    def get(self, key: str, *, tenant: str = "") -> CachedResponse | None:
        """Return the decrypted cached response for ``key``, else ``None`` (fails open)."""
        cipher = self._provider.for_tenant(tenant)
        if cipher is None:
            return None  # tenant shredded — key withheld, data inaccessible
        hit = self._inner.get(key, tenant=tenant)
        if hit is None:
            return None
        plaintext = cipher.open(key, hit.body)
        if plaintext is None:
            return None  # undecryptable entry -> treat as a miss
        return CachedResponse(
            status_code=hit.status_code, body=plaintext, content_type=hit.content_type
        )

    def put(self, key: str, value: CachedResponse, ttl_seconds: int, *, tenant: str = "") -> None:
        """Encrypt the body and store via the inner cache (fails open; never raises)."""
        cipher = self._provider.for_tenant(tenant)
        if cipher is None:
            return  # tenant shredded — do not cache
        try:
            sealed = cipher.seal(key, value.body)
        except Exception:
            return
        self._inner.put(
            key,
            CachedResponse(
                status_code=value.status_code, body=sealed, content_type=value.content_type
            ),
            ttl_seconds,
            tenant=tenant,
        )

    def close(self) -> None:
        """Close the inner cache if it supports it."""
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
