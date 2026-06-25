"""Data model for the memory graph (Track B/C of the graph-memory plan).

A small property graph of text-bearing nodes (turns/facts/entities) connected by typed edges.
Kept dependency-free and pure so the retrieval logic is trivially testable; persistence
(SQLite) and vector edges are added in later slices behind the same model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = ["MemoryEdge", "MemoryKind", "MemoryNode", "node_id"]


class MemoryKind(StrEnum):
    """The kind of a memory node."""

    TURN = "turn"  # a conversation turn (a message)
    FACT = "fact"  # a durable extracted statement
    ENTITY = "entity"  # a named thing (file, symbol, person, …)
    DECISION = "decision"  # a recorded decision


def node_id(kind: MemoryKind, text: str) -> str:
    """Return a stable id for a node from its kind and text (content-addressed)."""
    digest = hashlib.sha256(f"{kind.value}\x00{text}".encode()).hexdigest()
    return f"{kind.value}:{digest[:16]}"


@dataclass(frozen=True, slots=True)
class MemoryNode:
    """A text-bearing node in the memory graph.

    Args:
        id: Stable, content-addressed identifier (see :func:`node_id`).
        kind: The node kind.
        text: The node's text content.
        terms: Normalised content terms used for lexical retrieval.
    """

    id: str
    kind: MemoryKind
    text: str
    terms: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    """A typed, directed edge between two nodes.

    Args:
        src: Source node id.
        dst: Destination node id.
        kind: Relationship label (e.g. ``follows``, ``mentions``).
    """

    src: str
    dst: str
    kind: str
