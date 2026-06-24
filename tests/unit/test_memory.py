"""Tests for the graph-memory foundation (model, store, terms, ingest, retrieval)."""

from __future__ import annotations

from parsimony.memory import (
    GraphMemory,
    InMemoryGraphStore,
    MemoryEdge,
    MemoryKind,
    MemoryNode,
    extract_terms,
    ingest_request,
    jaccard,
    node_id,
    relevant,
)
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span
from parsimony.ports import MemoryPort


def _req(*texts: str, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=tuple(Message(role=Role.USER, spans=(Span(t),)) for t in texts),
        system=system,
    )


class TestModelAndTerms:
    def test_node_id_is_stable_and_content_addressed(self) -> None:
        assert node_id(MemoryKind.TURN, "hello") == node_id(MemoryKind.TURN, "hello")
        assert node_id(MemoryKind.TURN, "a") != node_id(MemoryKind.FACT, "a")

    def test_extract_terms_drops_stopwords_and_short_tokens(self) -> None:
        terms = extract_terms("The parser handles empty input")
        assert "parser" in terms
        assert "handles" in terms
        assert "the" not in terms  # stopword
        assert "input" in terms


class TestStore:
    def test_add_and_get_nodes_and_edges(self) -> None:
        store = InMemoryGraphStore()
        a = MemoryNode(id="a", kind=MemoryKind.TURN, text="x", terms=frozenset())
        b = MemoryNode(id="b", kind=MemoryKind.TURN, text="y", terms=frozenset())
        store.add_node(a)
        store.add_node(b)
        store.add_edge(MemoryEdge(src="a", dst="b", kind="follows"))
        assert store.get("a") is a
        assert store.node_count == 2
        assert store.edge_count == 1
        assert store.neighbors("a")[0].dst == "b"
        assert store.get("missing") is None


class TestJaccard:
    def test_similarity(self) -> None:
        assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0
        assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1 / 3
        assert jaccard(frozenset(), frozenset({"a"})) == 0.0


class TestIngest:
    def test_creates_turn_nodes_and_follows_edges(self) -> None:
        store = InMemoryGraphStore()
        ids = ingest_request(store, _req("first turn here", "second turn there"))
        assert len(ids) == 2
        assert store.node_count == 2
        assert store.edge_count == 1  # one 'follows' edge between the two turns

    def test_system_becomes_a_fact_node(self) -> None:
        store = InMemoryGraphStore()
        ids = ingest_request(store, _req("ask something", system="be a careful reviewer"))
        assert store.get(ids[0]).kind is MemoryKind.FACT
        assert store.node_count == 2

    def test_empty_messages_are_skipped(self) -> None:
        store = InMemoryGraphStore()
        ids = ingest_request(store, _req("   ", "real content"))
        assert len(ids) == 1


class TestRetrieval:
    def test_returns_most_relevant_first(self) -> None:
        store = InMemoryGraphStore()
        ingest_request(store, _req("the database migration failed", "lunch plans for friday"))
        hits = relevant(store, extract_terms("database migration error"))
        assert hits
        assert "database migration" in hits[0][0].text

    def test_zero_overlap_excluded(self) -> None:
        store = InMemoryGraphStore()
        ingest_request(store, _req("completely unrelated topic"))
        assert relevant(store, extract_terms("xyzzy quux")) == []

    def test_limit_is_respected(self) -> None:
        store = InMemoryGraphStore()
        ingest_request(store, _req("alpha beta", "alpha gamma", "alpha delta"))
        assert len(relevant(store, extract_terms("alpha"), limit=2)) == 2


class TestGraphMemory:
    def test_implements_memory_port(self) -> None:
        assert isinstance(GraphMemory(), MemoryPort)

    def test_ingest_then_relevant_returns_texts(self) -> None:
        memory = GraphMemory()
        memory.ingest(_req("optimize the token cache", "unrelated weather chat"))
        results = memory.relevant("token cache strategy", limit=1)
        assert results
        assert "token cache" in results[0]

    def test_relevant_on_empty_memory_is_empty(self) -> None:
        assert GraphMemory().relevant("anything") == ()
