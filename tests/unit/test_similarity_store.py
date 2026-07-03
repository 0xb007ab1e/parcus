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
