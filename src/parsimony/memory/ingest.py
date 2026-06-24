"""Ingest a canonical request into the memory graph (model-free, first slice).

This slice creates one node per non-empty message (and one for the system prompt) with
lexically-extracted terms, linking consecutive turns with ``follows`` edges. Smarter
extraction (entities/decisions, or a local model) plugs in behind the same shape later.
"""

from __future__ import annotations

from parsimony.memory.model import MemoryEdge, MemoryKind, MemoryNode, node_id
from parsimony.memory.store import GraphStore
from parsimony.memory.terms import extract_terms
from parsimony.model import CanonicalRequest

__all__ = ["ingest_request"]


def _add(store: GraphStore, kind: MemoryKind, text: str) -> str:
    identifier = node_id(kind, text)
    store.add_node(MemoryNode(id=identifier, kind=kind, text=text, terms=extract_terms(text)))
    return identifier


def ingest_request(store: GraphStore, request: CanonicalRequest) -> tuple[str, ...]:
    """Add nodes/edges for ``request`` to ``store`` and return the created node ids.

    Args:
        store: The graph store to populate.
        request: The canonical request to ingest.

    Returns:
        The ids of the nodes created (system node first if present, then turns in order).
    """
    ids: list[str] = []
    if request.system and request.system.strip():
        ids.append(_add(store, MemoryKind.FACT, request.system.strip()))

    previous: str | None = None
    for message in request.messages:
        text = message.text.strip()
        if not text:
            continue
        identifier = _add(store, MemoryKind.TURN, text)
        if previous is not None:
            store.add_edge(MemoryEdge(src=previous, dst=identifier, kind="follows"))
        previous = identifier
        ids.append(identifier)
    return tuple(ids)
