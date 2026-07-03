"""Durable snapshot for the similarity index (opt-in persistence, ``0600``).

The similarity index (:class:`~parcus.cache.similarity.SimilarityCache`) is otherwise in-memory
and cold after every restart. This module persists it as a **snapshot** on a sidecar SQLite file
so near-duplicate hits survive a restart, while the index keeps operating **in memory** on the hot
path — disk I/O happens only at startup (hydrate) and on write-through (``remember``), never inside
``lookup``.

Like the exact cache (:class:`~parcus.cache.sqlite_cache.SqliteCache`) the store is
**confidential** — a :class:`~parcus.cache.similarity.SimilarityEntry` holds a prompt-derived
embedding vector, never prompt text — so the backing file is created ``0600``. At-rest encryption
of the vector blob (parity with the exact cache) is layered on in a later slice; this slice
persists plaintext at ``0600``, mirroring the exact cache's posture when encryption is off.

Every operation **fails open**: a load/append error yields an empty snapshot / no-op rather than
raising, so a broken store degrades the index to in-memory-only without touching availability.
"""

from __future__ import annotations

import array
import os
import sqlite3
import threading

from parcus.cache.clock import SystemClock
from parcus.cache.similarity import SimilarityEntry
from parcus.ports import ClockPort

__all__ = ["SqliteSimilarityStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT NOT NULL,
    model      TEXT,
    tenant     TEXT NOT NULL,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL
)
"""
# float64 ('d'): exact round-trip of the embedder's Python floats, so cosine/threshold don't drift.
_VECTOR_TYPECODE = "d"


def _encode_vector(vector: list[float]) -> bytes:
    """Pack a float vector into a compact BLOB (float64, exact round-trip)."""
    return array.array(_VECTOR_TYPECODE, vector).tobytes()


def _decode_vector(blob: bytes) -> list[float]:
    """Unpack a BLOB produced by :func:`_encode_vector` back into a float list."""
    buf = array.array(_VECTOR_TYPECODE)
    buf.frombytes(blob)
    return list(buf)


class SqliteSimilarityStore:
    """A confidential, FIFO-bounded SQLite snapshot of the similarity index.

    Implements :class:`~parcus.cache.similarity.SimilarityStore`.

    Args:
        path: Sidecar database path; ``":memory:"`` for an ephemeral in-process store.
        max_entries: Cap on persisted rows; oldest are evicted first (FIFO), matching the
            in-memory index bound.
        clock: Injected time source (defaults to :class:`SystemClock`) for the audit column.
    """

    def __init__(
        self, path: str = ":memory:", *, max_entries: int = 2048, clock: ClockPort | None = None
    ) -> None:
        """Open (or create) the store, ensure its schema, and lock down file permissions."""
        self._max = max_entries
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            try:
                os.chmod(path, 0o600)  # confidential store — owner-only
            except OSError:
                pass
        with self._conn:
            self._conn.execute(_SCHEMA)

    def load(self) -> list[SimilarityEntry]:
        """Return the most-recent ``max_entries`` persisted entries, oldest first (fails open)."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT key, model, tenant, vector FROM entries ORDER BY rowid DESC LIMIT ?",
                    (self._max,),
                ).fetchall()
            entries = [
                SimilarityEntry(
                    vector=_decode_vector(bytes(vector)), key=key, model=model, tenant=tenant
                )
                for key, model, tenant, vector in rows
            ]
            entries.reverse()  # DESC fetch -> return oldest first (chronological)
            return entries
        except Exception:
            # Fail open: a broken store degrades the index to in-memory-only.
            return []

    def append(self, entry: SimilarityEntry) -> None:
        """Persist ``entry`` and FIFO-evict rows beyond ``max_entries`` (no-op on error)."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO entries (key, model, tenant, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        entry.key,
                        entry.model,
                        entry.tenant,
                        _encode_vector(entry.vector),
                        self._clock.now(),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM entries WHERE rowid NOT IN "
                    "(SELECT rowid FROM entries ORDER BY rowid DESC LIMIT ?)",
                    (self._max,),
                )
        except Exception:
            # Fail open: a persistence failure must never break the request path.
            return

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Close the connection on GC — a backstop; deterministic cleanup is ``close()``."""
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
