"""Tests for the SQLite-backed persistent graph store."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from parsimony.memory import (
    GraphMemory,
    GraphStore,
    MemoryEdge,
    MemoryKind,
    MemoryNode,
    SqliteGraphStore,
    ingest_request,
)
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _node(text: str, terms: frozenset[str]) -> MemoryNode:
    return MemoryNode(id=f"turn:{text}", kind=MemoryKind.TURN, text=text, terms=terms)


def _req(*texts: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=tuple(Message(role=Role.USER, spans=(Span(t),)) for t in texts),
    )


class TestSqliteGraphStore:
    def test_satisfies_graph_store_protocol(self) -> None:
        assert isinstance(SqliteGraphStore(), GraphStore)

    def test_node_roundtrip_reconstructs_terms_and_kind(self) -> None:
        store = SqliteGraphStore()
        store.add_node(_node("the parser", frozenset({"parser", "tokens"})))
        got = store.get("turn:the parser")
        assert got is not None
        assert got.kind is MemoryKind.TURN
        assert got.text == "the parser"
        assert got.terms == frozenset({"parser", "tokens"})

    def test_replace_is_idempotent_by_id(self) -> None:
        store = SqliteGraphStore()
        store.add_node(_node("x", frozenset({"a"})))
        store.add_node(_node("x", frozenset({"a"})))
        assert store.node_count == 1

    def test_edges_and_neighbors(self) -> None:
        store = SqliteGraphStore()
        store.add_node(_node("a", frozenset()))
        store.add_node(_node("b", frozenset()))
        store.add_edge(MemoryEdge(src="turn:a", dst="turn:b", kind="follows"))
        assert store.edge_count == 1
        assert store.neighbors("turn:a")[0].dst == "turn:b"
        assert store.neighbors("turn:b") == ()

    def test_get_missing_returns_none(self) -> None:
        assert SqliteGraphStore().get("nope") is None

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mem.sqlite")
        first = SqliteGraphStore(path)
        ingest_request(first, _req("durable knowledge here"))
        first.close()

        reopened = SqliteGraphStore(path)
        assert reopened.node_count == 1
        assert reopened.nodes()[0].text == "durable knowledge here"
        reopened.close()

    def test_database_file_is_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "mem.sqlite"
        store = SqliteGraphStore(str(path))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        store.close()


class TestGraphMemoryOverSqlite:
    def test_retrieval_works_over_persistent_store(self) -> None:
        memory = GraphMemory(SqliteGraphStore())
        memory.ingest(_req("optimize the token cache", "unrelated weather"))
        results = memory.relevant("token cache strategy", limit=1)
        assert results
        assert "token cache" in results[0]
