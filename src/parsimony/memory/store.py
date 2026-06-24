"""An in-memory property-graph store (dependency-free foundation).

Later slices add a SQLite + sqlite-vec backed store behind the same shape; the lexical
retrieval logic does not change.
"""

from __future__ import annotations

from parsimony.memory.model import MemoryEdge, MemoryNode

__all__ = ["InMemoryGraphStore"]


class InMemoryGraphStore:
    """A simple in-process graph of :class:`MemoryNode`s and :class:`MemoryEdge`s."""

    def __init__(self) -> None:
        """Start with an empty graph."""
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []

    def add_node(self, node: MemoryNode) -> None:
        """Insert or replace a node by id (idempotent for content-addressed ids)."""
        self._nodes[node.id] = node

    def add_edge(self, edge: MemoryEdge) -> None:
        """Add a directed edge (both endpoints should already exist)."""
        self._edges.append(edge)

    def get(self, node_id: str) -> MemoryNode | None:
        """Return the node with ``node_id`` if present."""
        return self._nodes.get(node_id)

    def nodes(self) -> tuple[MemoryNode, ...]:
        """Return all nodes (insertion order)."""
        return tuple(self._nodes.values())

    def neighbors(self, node_id: str) -> tuple[MemoryEdge, ...]:
        """Return edges originating at ``node_id``."""
        return tuple(edge for edge in self._edges if edge.src == node_id)

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)
