"""Unit tests for the persistent similarity-index snapshot (``SqliteSimilarityStore``)."""

from __future__ import annotations

import os
from pathlib import Path

from parcus.cache import SimilarityEntry, SqliteSimilarityStore


class _FakeClock:
    """A fixed time source for asserting the audit column."""

    def __init__(self, value: float) -> None:
        self._value = value

    def now(self) -> float:
        return self._value


def _entry(
    key: str, *, vector: list[float] | None = None, model: str | None = "m"
) -> SimilarityEntry:
    return SimilarityEntry(vector=vector or [1.0, 0.0], key=key, model=model, tenant="")


class TestSqliteSimilarityStore:
    def test_round_trips_entries_oldest_first(self) -> None:
        store = SqliteSimilarityStore()
        store.append(_entry("K1", vector=[1.0, 0.0]))
        store.append(_entry("K2", vector=[0.0, 1.0], model=None))
        loaded = store.load()
        assert [e.key for e in loaded] == ["K1", "K2"]  # chronological (oldest first)
        assert loaded[0].vector == [1.0, 0.0] and loaded[0].model == "m"
        assert loaded[1].vector == [0.0, 1.0] and loaded[1].model is None  # NULL model round-trips

    def test_vector_round_trips_exactly(self) -> None:
        # float64 packing must not drift, or cosine/threshold decisions would change on reload.
        vec = [0.123456789012345, -0.98765432109876, 3.0, 0.0]
        store = SqliteSimilarityStore()
        store.append(_entry("K1", vector=vec))
        assert store.load()[0].vector == vec

    def test_fifo_cap_keeps_most_recent(self) -> None:
        store = SqliteSimilarityStore(max_entries=2)
        for k in ("K1", "K2", "K3"):
            store.append(_entry(k))
        assert [e.key for e in store.load()] == ["K2", "K3"]  # K1 evicted

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = str(tmp_path / "sim.sqlite")
        first = SqliteSimilarityStore(path)
        first.append(_entry("K1"))
        first.close()
        assert [e.key for e in SqliteSimilarityStore(path).load()] == ["K1"]

    def test_backing_file_is_owner_only(self, tmp_path: Path) -> None:
        path = str(tmp_path / "sim.sqlite")
        SqliteSimilarityStore(path)
        assert oct(os.stat(path).st_mode & 0o777) == oct(0o600)

    def test_load_fails_open(self) -> None:
        store = SqliteSimilarityStore()
        store.close()  # subsequent queries raise -> must degrade to empty
        assert store.load() == []

    def test_append_fails_open(self) -> None:
        store = SqliteSimilarityStore()
        store.close()
        store.append(_entry("K1"))  # no raise

    def test_uses_injected_clock_for_created_at(self) -> None:
        store = SqliteSimilarityStore(clock=_FakeClock(123.0))
        store.append(_entry("K1"))
        # created_at isn't exposed by the API (ordering is by rowid); assert it via the store conn.
        assert store._conn.execute("SELECT created_at FROM entries").fetchone()[0] == 123.0


class TestSqliteSimilarityStoreEncryption:
    """At-rest vector encryption via a shared CipherProvider (parity with the exact cache)."""

    @staticmethod
    def _provider(shredded: frozenset[str] = frozenset(), *, master: bytes = b"\x11" * 32):
        from parcus.cache.encryption import TenantCipherProvider

        return TenantCipherProvider(master, shredded=shredded)

    def test_vector_is_sealed_at_rest(self) -> None:
        vec = [0.5, -0.25]
        store = SqliteSimilarityStore(provider=self._provider())
        store.append(SimilarityEntry(vector=vec, key="K1", model="m", tenant="t"))
        stored = store._conn.execute("SELECT vector FROM entries").fetchone()[0]
        from parcus.cache.similarity_store import _encode_vector

        assert bytes(stored) != _encode_vector(vec)  # not plaintext floats
        assert bytes(stored)[:1] == b"\x01"  # sealed blob carries the version byte

    def test_encrypted_round_trip(self) -> None:
        vec = [0.123456789012345, -0.98765432109876]
        provider = self._provider()
        store = SqliteSimilarityStore(provider=provider)
        store.append(SimilarityEntry(vector=vec, key="K1", model="m", tenant="t"))
        loaded = store.load()
        assert len(loaded) == 1 and loaded[0].vector == vec  # AEAD open + exact float round-trip

    def test_encrypted_round_trip_across_instances(self, tmp_path: Path) -> None:
        path = str(tmp_path / "sim.sqlite")
        SqliteSimilarityStore(path, provider=self._provider()).append(
            SimilarityEntry(vector=[1.0, 2.0], key="K1", model="m", tenant="t")
        )
        reopened = SqliteSimilarityStore(path, provider=self._provider()).load()
        assert [e.key for e in reopened] == ["K1"] and reopened[0].vector == [1.0, 2.0]

    def test_shredded_tenant_not_persisted_on_append(self) -> None:
        store = SqliteSimilarityStore(provider=self._provider(shredded=frozenset({"t"})))
        store.append(SimilarityEntry(vector=[1.0, 0.0], key="K1", model="m", tenant="t"))
        assert store._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0

    def test_shredded_tenant_row_skipped_on_load(self) -> None:
        # Written normally, then a later provider shreds tenant "t" -> its rows drop on load.
        store = SqliteSimilarityStore(provider=self._provider())
        store.append(SimilarityEntry(vector=[1.0, 0.0], key="A", model="m", tenant="t"))
        store.append(SimilarityEntry(vector=[0.0, 1.0], key="B", model="m", tenant="keep"))
        store._provider = self._provider(shredded=frozenset({"t"}))
        assert [e.key for e in store.load()] == ["B"]  # "A" (tenant t) inaccessible

    def test_wrong_key_row_skipped_on_load(self) -> None:
        store = SqliteSimilarityStore(provider=self._provider(master=b"\x11" * 32))
        store.append(SimilarityEntry(vector=[1.0, 0.0], key="A", model="m", tenant="t"))
        store._provider = self._provider(master=b"\x22" * 32)  # different master -> can't open
        assert store.load() == []

    def test_plaintext_blob_skipped_when_provider_added(self) -> None:
        # Toggling encryption on over a plaintext file: old rows don't open -> skipped, not garbage.
        store = SqliteSimilarityStore()  # plaintext write
        store.append(SimilarityEntry(vector=[1.0, 0.0], key="A", model="m", tenant="t"))
        store._provider = self._provider()  # now expects sealed blobs
        assert store.load() == []
