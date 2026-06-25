"""Unit tests for the opt-in semantic (near-duplicate) cache index."""

from __future__ import annotations

from collections.abc import Sequence

from parcus.cache import SimilarityCache


class _StubEmbedder:
    """Maps preset texts to fixed vectors so similarity is deterministic in tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


# Three unit vectors: a and a2 are near-identical; b is orthogonal.
_VECS = {
    "a": [1.0, 0.0],
    "a2": [0.99, 0.14],  # ~0.99 cosine with a
    "b": [0.0, 1.0],  # 0 cosine with a
}


def _cache(threshold: float = 0.97, max_entries: int = 2048) -> SimilarityCache:
    return SimilarityCache(_StubEmbedder(_VECS), threshold=threshold, max_entries=max_entries)


def test_near_duplicate_above_threshold_hits() -> None:
    cache = _cache()
    cache.remember(text="a", key="K1", model="m", tenant="")
    assert cache.lookup(text="a2", model="m", tenant="") == "K1"


def test_dissimilar_below_threshold_misses() -> None:
    cache = _cache()
    cache.remember(text="a", key="K1", model="m", tenant="")
    assert cache.lookup(text="b", model="m", tenant="") is None


def test_different_model_never_matches() -> None:
    cache = _cache()
    cache.remember(text="a", key="K1", model="claude", tenant="")
    assert cache.lookup(text="a2", model="gpt", tenant="") is None


def test_different_tenant_never_matches() -> None:
    # Cross-tenant similar-serve would be threat E1 — must not happen.
    cache = _cache()
    cache.remember(text="a", key="K1", model="m", tenant="tenant-a")
    assert cache.lookup(text="a2", model="m", tenant="tenant-b") is None
    assert cache.lookup(text="a2", model="m", tenant="tenant-a") == "K1"


def test_empty_index_misses() -> None:
    assert _cache().lookup(text="a", model="m", tenant="") is None


def test_fifo_eviction_caps_entries() -> None:
    cache = _cache(max_entries=1)
    cache.remember(text="a", key="K1", model="m", tenant="")
    cache.remember(text="b", key="K2", model="m", tenant="")  # evicts K1
    # K1's vector (a) is gone, so a near-duplicate of it no longer hits.
    assert cache.lookup(text="a2", model="m", tenant="") is None


def test_fails_open_on_embedder_error() -> None:
    class _Boom:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("boom")

    cache = SimilarityCache(_Boom())
    cache.remember(text="a", key="K1", model="m", tenant="")  # no raise
    assert cache.lookup(text="a", model="m", tenant="") is None  # no raise, no hit
