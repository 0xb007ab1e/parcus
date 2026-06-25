"""Tests for at-rest cache encryption (AES-256-GCM cipher + EncryptedCache decorator)."""

from __future__ import annotations

import os

import pytest

from parsimony.cache import SqliteCache
from parsimony.cache.encryption import (
    CacheCipher,
    EncryptedCache,
    StaticCipherProvider,
    TenantCipherProvider,
)
from parsimony.model import CachedResponse

_KEY = b"\x00" * 32
_KEY2 = b"\x01" * 32


class TestCacheCipher:
    def test_round_trip(self) -> None:
        cipher = CacheCipher(_KEY)
        blob = cipher.seal("k", b"secret bytes")
        assert cipher.open("k", blob) == b"secret bytes"

    def test_ciphertext_hides_plaintext(self) -> None:
        blob = CacheCipher(_KEY).seal("k", b"the secret answer")
        assert b"the secret answer" not in blob

    def test_unique_nonce_per_seal(self) -> None:
        cipher = CacheCipher(_KEY)
        assert cipher.seal("k", b"x") != cipher.seal("k", b"x")  # random nonce each time

    def test_rejects_wrong_key_length(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            CacheCipher(b"too short")

    def test_tamper_detected(self) -> None:
        cipher = CacheCipher(_KEY)
        blob = bytearray(cipher.seal("k", b"data"))
        blob[-1] ^= 0x01  # flip a ciphertext/tag bit
        assert cipher.open("k", bytes(blob)) is None

    def test_wrong_key_fails(self) -> None:
        blob = CacheCipher(_KEY).seal("k", b"data")
        assert CacheCipher(_KEY2).open("k", blob) is None

    def test_aad_binds_to_cache_key(self) -> None:
        # A ciphertext sealed for one key cannot be opened under another (no relocation).
        cipher = CacheCipher(_KEY)
        blob = cipher.seal("key-A", b"data")
        assert cipher.open("key-B", blob) is None

    def test_unknown_version_or_short_blob_is_none(self) -> None:
        cipher = CacheCipher(_KEY)
        assert cipher.open("k", b"\x99" + b"\x00" * 20) is None  # bad version byte
        assert cipher.open("k", b"\x01") is None  # too short for nonce

    def test_rejects_wrong_length_previous_key(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            CacheCipher(_KEY, previous_keys=(b"short",))


class TestKeyRotation:
    def test_seals_with_current_key_only(self) -> None:
        # After rotation, new writes use the current key; the old key alone can't read them.
        rotated = CacheCipher(_KEY2, previous_keys=(_KEY,))
        blob = rotated.seal("k", b"new data")
        assert CacheCipher(_KEY).open("k", blob) is None  # old key can't open new entry
        assert rotated.open("k", blob) == b"new data"  # current key opens it

    def test_opens_entry_sealed_under_previous_key(self) -> None:
        # An entry sealed before rotation (under _KEY) stays readable after rotating to _KEY2.
        old_blob = CacheCipher(_KEY).seal("k", b"old data")
        rotated = CacheCipher(_KEY2, previous_keys=(_KEY,))
        assert rotated.open("k", old_blob) == b"old data"

    def test_unknown_key_still_fails_after_rotation(self) -> None:
        blob = CacheCipher(b"\x02" * 32).seal("k", b"data")  # sealed under a third, dropped key
        rotated = CacheCipher(_KEY2, previous_keys=(_KEY,))
        assert rotated.open("k", blob) is None


class TestEncryptedCache:
    def test_stores_ciphertext_and_round_trips(self) -> None:
        inner = SqliteCache()
        enc = EncryptedCache(inner, CacheCipher(os.urandom(32)))
        enc.put(
            "k",
            CachedResponse(status_code=200, body=b"plain body", content_type="application/json"),
            60,
        )
        # The inner store holds ciphertext, not the plaintext body.
        raw = inner.get("k")
        assert raw is not None
        assert raw.body != b"plain body"
        assert b"plain body" not in raw.body
        # The decorator decrypts on the way out.
        got = enc.get("k")
        assert got is not None
        assert got.body == b"plain body"
        assert got.status_code == 200
        assert got.content_type == "application/json"

    def test_miss_returns_none(self) -> None:
        enc = EncryptedCache(SqliteCache(), CacheCipher(_KEY))
        assert enc.get("absent") is None

    def test_undecryptable_entry_is_a_miss(self) -> None:
        # An inner entry that isn't valid ciphertext (e.g. wrong key / legacy plaintext) -> miss.
        inner = SqliteCache()
        inner.put(
            "k", CachedResponse(status_code=200, body=b"not ciphertext", content_type=None), 60
        )
        enc = EncryptedCache(inner, CacheCipher(_KEY))
        assert enc.get("k") is None

    def test_different_key_cannot_read(self) -> None:
        inner = SqliteCache()
        EncryptedCache(inner, CacheCipher(_KEY)).put(
            "k", CachedResponse(status_code=200, body=b"secret", content_type=None), 60
        )
        assert EncryptedCache(inner, CacheCipher(_KEY2)).get("k") is None

    def test_close_delegates_to_inner(self) -> None:
        inner = SqliteCache()
        EncryptedCache(inner, CacheCipher(_KEY)).close()
        # After close, the inner connection is closed; a get fails open to None.
        assert inner.get("anything") is None

    def test_put_fails_open_when_seal_raises(self) -> None:
        class _BoomCipher:
            def seal(self, aad: str, plaintext: bytes) -> bytes:
                raise RuntimeError("boom")

            def open(self, aad: str, blob: bytes) -> bytes | None:
                return None

        inner = SqliteCache()
        enc = EncryptedCache(inner, _BoomCipher())  # type: ignore[arg-type]
        enc.put("k", CachedResponse(status_code=200, body=b"x", content_type=None), 60)  # no raise
        assert inner.get("k") is None  # nothing was stored

    def test_survives_key_rotation(self) -> None:
        # Write under the old key, then rotate: the entry is still served via the previous key.
        inner = SqliteCache()
        EncryptedCache(inner, CacheCipher(_KEY)).put(
            "k", CachedResponse(status_code=200, body=b"pre-rotation", content_type=None), 60
        )
        rotated = EncryptedCache(inner, CacheCipher(_KEY2, previous_keys=(_KEY,)))
        got = rotated.get("k")
        assert got is not None
        assert got.body == b"pre-rotation"

    def test_close_is_noop_when_inner_has_no_close(self) -> None:
        class _NoCloseCache:
            def get(self, key: str) -> CachedResponse | None:
                return None

            def put(self, key: str, value: CachedResponse, ttl_seconds: int) -> None:
                return None

        EncryptedCache(_NoCloseCache(), CacheCipher(_KEY)).close()  # no raise


def _resp(body: bytes = b"secret") -> CachedResponse:
    return CachedResponse(status_code=200, body=body, content_type=None)


class TestTenantCipherProvider:
    def test_distinct_dek_per_tenant(self) -> None:
        provider = TenantCipherProvider(_KEY)
        # A blob sealed for tenant a can't be opened with tenant b's cipher (distinct DEKs).
        a, b = provider.for_tenant("a"), provider.for_tenant("b")
        assert a is not None and b is not None
        blob = a.seal("k", b"data")
        assert b.open("k", blob) is None
        assert a.open("k", blob) == b"data"

    def test_same_tenant_same_cipher(self) -> None:
        provider = TenantCipherProvider(_KEY)
        assert provider.for_tenant("a") is provider.for_tenant("a")  # cached

    def test_shredded_tenant_has_no_key(self) -> None:
        provider = TenantCipherProvider(_KEY, shredded=frozenset({"gone"}))
        assert provider.for_tenant("gone") is None
        assert provider.for_tenant("present") is not None

    def test_master_rotation_derives_previous_dek(self) -> None:
        # An entry sealed under the old master's per-tenant DEK opens after rotating the master.
        old_blob = TenantCipherProvider(_KEY).for_tenant("t").seal("k", b"old")  # type: ignore[union-attr]
        rotated = TenantCipherProvider(_KEY2, previous_master_keys=(_KEY,))
        assert rotated.for_tenant("t").open("k", old_blob) == b"old"  # type: ignore[union-attr]

    def test_rejects_wrong_length_master(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            TenantCipherProvider(b"short")


class TestEncryptedCacheWithProvider:
    def test_requires_exactly_one_of_cipher_or_provider(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            EncryptedCache(SqliteCache())  # neither
        with pytest.raises(ValueError, match="exactly one"):
            EncryptedCache(
                SqliteCache(), CacheCipher(_KEY), provider=StaticCipherProvider(CacheCipher(_KEY))
            )

    def test_per_tenant_round_trip(self) -> None:
        enc = EncryptedCache(SqliteCache(), provider=TenantCipherProvider(_KEY))
        enc.put("k", _resp(b"tenant-a data"), 60, tenant="a")
        got = enc.get("k", tenant="a")
        assert got is not None and got.body == b"tenant-a data"

    def test_shredded_tenant_reads_miss_and_writes_skip(self) -> None:
        inner = SqliteCache()
        # Tenant 'gone' caches normally...
        live = EncryptedCache(inner, provider=TenantCipherProvider(_KEY))
        live.put("k", _resp(b"to be erased"), 60, tenant="gone")
        assert live.get("k", tenant="gone") is not None
        # ...then is shredded: its key is withheld -> existing entry inaccessible, new writes skip.
        shredded = EncryptedCache(
            inner, provider=TenantCipherProvider(_KEY, shredded=frozenset({"gone"}))
        )
        assert shredded.get("k", tenant="gone") is None  # erased
        shredded.put("k2", _resp(b"nope"), 60, tenant="gone")
        assert inner.get("k2") is None  # nothing written for a shredded tenant

    def test_one_tenant_cannot_decrypt_anothers_entry(self) -> None:
        # Even at the same cache key, tenant b's DEK can't open tenant a's ciphertext.
        inner = SqliteCache()
        enc = EncryptedCache(inner, provider=TenantCipherProvider(_KEY))
        enc.put("shared-key", _resp(b"a-only"), 60, tenant="a")
        assert enc.get("shared-key", tenant="b") is None
        assert enc.get("shared-key", tenant="a").body == b"a-only"  # type: ignore[union-attr]
