"""Tests for at-rest cache encryption (AES-256-GCM cipher + EncryptedCache decorator)."""

from __future__ import annotations

import os

import pytest

from parsimony.cache import SqliteCache
from parsimony.cache.encryption import CacheCipher, EncryptedCache
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

    def test_close_is_noop_when_inner_has_no_close(self) -> None:
        class _NoCloseCache:
            def get(self, key: str) -> CachedResponse | None:
                return None

            def put(self, key: str, value: CachedResponse, ttl_seconds: int) -> None:
                return None

        EncryptedCache(_NoCloseCache(), CacheCipher(_KEY)).close()  # no raise
