"""Embedders for semantic retrieval — local-only, lazy, opt-in.

Two adapters behind one :class:`EmbedderPort`:

* :class:`HashingEmbedder` — deterministic, dependency-free feature-hashing vectors. No model,
  no network; exercises the whole vectorise → cosine → top-k pipeline and is what the test
  suite uses. (It is lexical in nature, so it does not capture true synonymy — it exists to make
  the pipeline real and testable, not to be the production semantic model.)
* :class:`SentenceTransformerEmbedder` — the real **local** semantic model, lazily imported from
  the optional ``embeddings`` extra. Never makes a network call at inference; the model is
  loaded from local cache. Not imported by the test suite.

A memory that phoned a remote embedding API would defeat the project's purpose, so there is no
remote embedder.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from parsimony.memory.terms import extract_terms

__all__ = ["EmbedderPort", "HashingEmbedder", "SentenceTransformerEmbedder", "cosine"]


@runtime_checkable
class EmbedderPort(Protocol):
    """Maps texts to dense vectors (local, deterministic per implementation)."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two equal-length vectors (0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder. Implements ``EmbedderPort``.

    Hashes each content term into a fixed-dimension vector with a signed bucket, then L2-
    normalises. Reproducible (uses SHA-256, not the salted built-in ``hash``).

    Args:
        dim: Vector dimensionality.
    """

    def __init__(self, dim: int = 256) -> None:
        """Initialise with the embedding dimensionality."""
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one hashed, L2-normalised vector per text."""
        return [self._vectorise(text) for text in texts]

    def _vectorise(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for term in extract_terms(text):
            digest = int.from_bytes(hashlib.sha256(term.encode()).digest()[:8], "big")
            bucket = digest % self._dim
            sign = 1.0 if (digest >> 17) & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0.0:
            vector = [x / norm for x in vector]
        return vector


class SentenceTransformerEmbedder:
    """Real local semantic embedder (lazy import of the optional ``embeddings`` extra).

    Args:
        model_name: A sentence-transformers model available in the local cache.

    Raises:
        ImportError: If the ``embeddings`` extra is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Lazily load the local model; never performs a network call at inference."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
            raise ImportError(
                "local embeddings require the 'embeddings' extra: "
                "pip install 'parsimony[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - needs model
        """Encode ``texts`` with the local model."""
        return [[float(x) for x in row] for row in self._model.encode(list(texts))]
