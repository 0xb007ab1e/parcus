"""Tests for embedders and semantic retrieval (no model download — hermetic)."""

from __future__ import annotations

import importlib.util

import pytest

from parcus.memory import (
    EmbedderPort,
    GraphMemory,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine,
)
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*texts: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=tuple(Message(role=Role.USER, spans=(Span(t),)) for t in texts),
    )


class TestCosine:
    def test_identical_is_one(self) -> None:
        assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_is_zero(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_is_zero(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestHashingEmbedder:
    def test_dimension_and_determinism(self) -> None:
        emb = HashingEmbedder(dim=64)
        v1 = emb.embed(["the token cache strategy"])[0]
        v2 = emb.embed(["the token cache strategy"])[0]
        assert len(v1) == 64
        assert v1 == v2  # deterministic (SHA-256 based, not salted hash())

    def test_normalised(self) -> None:
        (vector,) = HashingEmbedder().embed(["several distinct content terms here"])
        norm = sum(x * x for x in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_shared_terms_are_more_similar(self) -> None:
        emb = HashingEmbedder()
        a, b, c = emb.embed(
            [
                "database migration tenant column",
                "database migration tenant index",
                "weather report",
            ]
        )
        assert cosine(a, b) > cosine(a, c)

    def test_satisfies_embedder_port(self) -> None:
        assert isinstance(HashingEmbedder(), EmbedderPort)

    def test_empty_text_yields_zero_vector(self) -> None:
        # No content terms -> zero norm -> normalisation is skipped (the else branch).
        (vector,) = HashingEmbedder(dim=8).embed([""])
        assert vector == [0.0] * 8


class _FakeEmbedder:
    """A deterministic 'semantic' embedder: maps known phrases to chosen vectors."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        return [self._table.get(t.strip(), [0.0, 0.0, 0.0]) for t in texts]


class TestSemanticRetrieval:
    def test_uses_embedder_when_provided(self) -> None:
        # "cars" and "automobiles" share no tokens but are placed close in vector space.
        table = {
            "automobiles are taxed annually": [1.0, 0.0, 0.0],
            "the weather is nice today": [0.0, 1.0, 0.0],
            "how are cars taxed": [0.95, 0.05, 0.0],
        }
        memory = GraphMemory(embedder=_FakeEmbedder(table))
        memory.ingest(_req("automobiles are taxed annually", "the weather is nice today"))
        results = memory.relevant("how are cars taxed", limit=1)
        assert results == ("automobiles are taxed annually",)  # semantic match, not lexical

    def test_falls_back_to_lexical_without_embedder(self) -> None:
        memory = GraphMemory()  # no embedder
        memory.ingest(_req("token cache design", "unrelated weather"))
        assert "token cache" in memory.relevant("token cache", limit=1)[0]

    def test_empty_semantic_memory_returns_empty(self) -> None:
        assert GraphMemory(embedder=_FakeEmbedder({})).relevant("anything") == ()


class TestSentenceTransformerEmbedder:
    def test_missing_extra_raises_helpful_error(self) -> None:
        if importlib.util.find_spec("sentence_transformers") is not None:
            pytest.skip("sentence-transformers is installed; skipping the absent-extra path")
        with pytest.raises(ImportError, match="embeddings"):
            SentenceTransformerEmbedder()
