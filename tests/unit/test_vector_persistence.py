"""Tests for persisted embedding vectors (semantic retrieval survives a restart)."""

from __future__ import annotations

from pathlib import Path

from parsimony.memory import GraphMemory, HashingEmbedder, InMemoryGraphStore, SqliteGraphStore
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*texts: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=tuple(Message(role=Role.USER, spans=(Span(t),)) for t in texts),
    )


class TestStoreVectors:
    def test_in_memory_roundtrip(self) -> None:
        store = InMemoryGraphStore()
        store.set_vector("a", [1.0, 2.0])
        assert store.get_vector("a") == [1.0, 2.0]
        assert store.get_vector("missing") is None

    def test_sqlite_roundtrip_and_missing(self) -> None:
        store = SqliteGraphStore()
        store.set_vector("a", [1.0, 2.0, 3.0])
        assert store.get_vector("a") == [1.0, 2.0, 3.0]
        assert store.get_vector("missing") is None

    def test_sqlite_replace(self) -> None:
        store = SqliteGraphStore()
        store.set_vector("a", [1.0])
        store.set_vector("a", [2.0])
        assert store.get_vector("a") == [2.0]


class TestPersistentSemanticRetrieval:
    def test_vectors_survive_restart(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mem.sqlite")
        embedder = HashingEmbedder()

        store = SqliteGraphStore(path)
        first = GraphMemory(store, embedder=embedder)
        first.ingest(_req("the database migration adds tenant_id", "weather is sunny"))
        store.close()

        # Reopen the persisted DB with NO re-ingest; semantic retrieval still works.
        reopened = SqliteGraphStore(path)
        second = GraphMemory(reopened, embedder=embedder)
        results = second.relevant("tenant_id migration", limit=1)
        assert results
        assert "tenant_id" in results[0]
        reopened.close()
