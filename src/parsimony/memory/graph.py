"""``GraphMemory`` — a :class:`parsimony.ports.MemoryPort` over the in-memory graph.

Provides ``ingest`` (add a request's content to the graph) and ``relevant`` (retrieve the most
relevant prior snippets for a query). Retrieval is **lexical by default**; pass an
:class:`~parsimony.memory.embedding.EmbedderPort` for **semantic** (cosine) retrieval. This is
the read-augmenting substrate for Track B (context retrieval) and Track C (compaction); it is
**not yet wired into the live request path** — injecting retrieved context into prompts is a
separate, eval-gated slice.
"""

from __future__ import annotations

from parsimony.memory.embedding import EmbedderPort, cosine
from parsimony.memory.ingest import ingest_request
from parsimony.memory.retrieval import relevant
from parsimony.memory.store import GraphStore, InMemoryGraphStore
from parsimony.memory.terms import extract_terms
from parsimony.model import CanonicalRequest

__all__ = ["GraphMemory"]


class GraphMemory:
    """Graph memory over any :class:`GraphStore`. Implements :class:`parsimony.ports.MemoryPort`.

    Args:
        store: The backing graph store (defaults to a fresh in-memory store).
        embedder: Optional local embedder; when given, ``relevant`` uses semantic (cosine)
            retrieval over node vectors, otherwise lexical (Jaccard) retrieval.
    """

    def __init__(
        self,
        store: GraphStore | None = None,
        embedder: EmbedderPort | None = None,
    ) -> None:
        """Initialise with an optional pre-existing store and optional embedder."""
        self._store: GraphStore = store or InMemoryGraphStore()
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}

    @property
    def store(self) -> GraphStore:
        """The backing graph store (for inspection/persistence)."""
        return self._store

    def ingest(self, request: CanonicalRequest) -> None:
        """Add the request's content to the graph (and index vectors if semantic)."""
        ingest_request(self._store, request)
        if self._embedder is not None:
            self._reindex(self._embedder)

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        """Return the texts of the nodes most relevant to ``query`` (semantic or lexical)."""
        if self._embedder is None:
            terms = extract_terms(query)
            return tuple(node.text for node, _ in relevant(self._store, terms, limit=limit))
        return self._semantic_relevant(self._embedder, query, limit)

    def _reindex(self, embedder: EmbedderPort) -> None:
        """Embed any store nodes that do not yet have a cached vector."""
        missing = [node for node in self._store.nodes() if node.id not in self._vectors]
        if not missing:
            return
        vectors = embedder.embed([node.text for node in missing])
        for node, vector in zip(missing, vectors, strict=True):
            self._vectors[node.id] = vector

    def _semantic_relevant(self, embedder: EmbedderPort, query: str, limit: int) -> tuple[str, ...]:
        query_vector = embedder.embed([query])[0]
        scored = [
            (node, cosine(query_vector, self._vectors[node.id]))
            for node in self._store.nodes()
            if node.id in self._vectors
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return tuple(node.text for node, score in scored[:limit] if score > 0.0)
