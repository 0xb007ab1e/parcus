"""Lexical relevance retrieval over the memory graph (no model required)."""

from __future__ import annotations

from parsimony.memory.model import MemoryNode
from parsimony.memory.store import GraphStore

__all__ = ["jaccard", "relevant"]


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Return the Jaccard similarity of two term sets, in ``[0.0, 1.0]``."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def relevant(
    store: GraphStore,
    query_terms: frozenset[str],
    *,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[tuple[MemoryNode, float]]:
    """Return the nodes most lexically relevant to ``query_terms``, best first.

    Args:
        store: The graph to search.
        query_terms: The query's extracted terms.
        limit: Maximum number of results.
        min_score: Exclusive lower bound on the similarity score.

    Returns:
        ``(node, score)`` pairs sorted by descending score then descending term overlap.
    """
    scored: list[tuple[MemoryNode, float]] = []
    for node in store.nodes():
        score = jaccard(query_terms, node.terms)
        if score > min_score:
            scored.append((node, score))
    scored.sort(key=lambda item: (item[1], len(item[0].terms & query_terms)), reverse=True)
    return scored[:limit]
