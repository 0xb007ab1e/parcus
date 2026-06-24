"""``GraphMemory`` — a :class:`parsimony.ports.MemoryPort` over the in-memory graph.

Provides ``ingest`` (add a request's content to the graph) and ``relevant`` (retrieve the most
relevant prior snippets for a query). This is the read-augmenting substrate for Track B
(context retrieval) and Track C (compaction); it is **not yet wired into the live request
path** — injecting retrieved context into prompts is a separate, eval-gated slice.
"""

from __future__ import annotations

from parsimony.memory.ingest import ingest_request
from parsimony.memory.retrieval import relevant
from parsimony.memory.store import InMemoryGraphStore
from parsimony.memory.terms import extract_terms
from parsimony.model import CanonicalRequest

__all__ = ["GraphMemory"]


class GraphMemory:
    """In-memory graph memory. Implements :class:`parsimony.ports.MemoryPort`.

    Args:
        store: The backing graph store (defaults to a fresh in-memory store).
    """

    def __init__(self, store: InMemoryGraphStore | None = None) -> None:
        """Initialise with an optional pre-existing store."""
        self._store = store or InMemoryGraphStore()

    @property
    def store(self) -> InMemoryGraphStore:
        """The backing graph store (for inspection/persistence)."""
        return self._store

    def ingest(self, request: CanonicalRequest) -> None:
        """Add the request's content to the graph."""
        ingest_request(self._store, request)

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        """Return the texts of the nodes most lexically relevant to ``query``."""
        terms = extract_terms(query)
        return tuple(node.text for node, _ in relevant(self._store, terms, limit=limit))
