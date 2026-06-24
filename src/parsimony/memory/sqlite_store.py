"""A SQLite-backed persistent graph store (implements :class:`GraphStore`).

Durable across proxy restarts; mirrors the cache's confidentiality posture (file created
``0600``). Terms are stored space-joined and reconstructed on read. Vector edges (sqlite-vec)
for semantic retrieval are a later slice.
"""

from __future__ import annotations

import os
import sqlite3
import threading

from parsimony.memory.model import MemoryEdge, MemoryKind, MemoryNode

__all__ = ["SqliteGraphStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id    TEXT PRIMARY KEY,
    kind  TEXT NOT NULL,
    text  TEXT NOT NULL,
    terms TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src  TEXT NOT NULL,
    dst  TEXT NOT NULL,
    kind TEXT NOT NULL
);
"""


def _to_node(row: tuple[str, str, str, str]) -> MemoryNode:
    identifier, kind, text, terms = row
    return MemoryNode(
        id=identifier,
        kind=MemoryKind(kind),
        text=text,
        terms=frozenset(terms.split()),
    )


class SqliteGraphStore:
    """A persistent graph store over SQLite. Implements :class:`GraphStore`.

    Args:
        path: Database path; ``":memory:"`` for an ephemeral store.
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the database, set owner-only permissions, and ensure the schema."""
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            try:
                os.chmod(path, 0o600)  # confidential store — owner-only
            except OSError:
                pass
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def add_node(self, node: MemoryNode) -> None:
        """Insert or replace a node by id."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes (id, kind, text, terms) VALUES (?, ?, ?, ?)",
                (node.id, node.kind.value, node.text, " ".join(sorted(node.terms))),
            )

    def add_edge(self, edge: MemoryEdge) -> None:
        """Add a directed edge."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO edges (src, dst, kind) VALUES (?, ?, ?)",
                (edge.src, edge.dst, edge.kind),
            )

    def get(self, node_id: str) -> MemoryNode | None:
        """Return the node with ``node_id`` if present."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, kind, text, terms FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
        return _to_node(row) if row is not None else None

    def nodes(self) -> tuple[MemoryNode, ...]:
        """Return all nodes."""
        with self._lock:
            rows = self._conn.execute("SELECT id, kind, text, terms FROM nodes").fetchall()
        return tuple(_to_node(row) for row in rows)

    def neighbors(self, node_id: str) -> tuple[MemoryEdge, ...]:
        """Return edges originating at ``node_id``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT src, dst, kind FROM edges WHERE src = ?", (node_id,)
            ).fetchall()
        return tuple(MemoryEdge(src=s, dst=d, kind=k) for s, d, k in rows)

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    @property
    def edge_count(self) -> int:
        """Number of edges."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
