"""Durable snapshot for the similarity index (opt-in persistence, ``0600``).

The similarity index (:class:`~parcus.cache.similarity.SimilarityCache`) is otherwise in-memory
and cold after every restart. This module persists it as a **snapshot** on a sidecar SQLite file
so near-duplicate hits survive a restart, while the index keeps operating **in memory** on the hot
path — disk I/O happens only at startup (hydrate) and on write-through (``remember``), never inside
``lookup``.

Like the exact cache (:class:`~parcus.cache.sqlite_cache.SqliteCache`) the store is
**confidential** — a :class:`~parcus.cache.similarity.SimilarityEntry` holds a prompt-derived
embedding vector, never prompt text — so the backing file is created ``0600``. When a
:class:`~parcus.cache.encryption.CipherProvider` is supplied (i.e. ``cache_encryption`` is on) the
vector blob is **sealed at rest with the same per-tenant cipher as the exact cache** — AAD-bound to
the entry's exact-cache key, with per-tenant DEKs and crypto-shred parity. With no provider it
persists plaintext at ``0600``, mirroring the exact cache's posture when encryption is off.

Every operation **fails open**: a load/append error yields an empty snapshot / no-op rather than
raising, so a broken store degrades the index to in-memory-only without touching availability. A
row that a shredded/rotated key can't open is skipped on load (crypto-shredding), not served.
"""

from __future__ import annotations

import array
import os
import sqlite3
import threading
from typing import TYPE_CHECKING

from parcus.cache.clock import SystemClock
from parcus.cache.similarity import SimilarityEntry
from parcus.ports import ClockPort

if TYPE_CHECKING:  # annotation only — keeps `cryptography` (the encryption extra) optional
    from parcus.cache.encryption import CipherProvider

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
        provider: Optional cipher provider; when set, vector blobs are sealed at rest with the
            tenant's cipher (AAD = the entry's exact-cache key). ``None`` = plaintext at ``0600``.
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        max_entries: int = 2048,
        clock: ClockPort | None = None,
        provider: CipherProvider | None = None,
    ) -> None:
        """Open (or create) the store, ensure its schema, and lock down file permissions."""
        self._max = max_entries
        self._clock = clock or SystemClock()
        self._provider = provider
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
        """Return the most-recent ``max_entries`` persisted entries, oldest first (fails open).

        Each row is decoded independently: a row a shredded/rotated key can't open, or whose blob
        is malformed, is **skipped** (not served, not fatal) — so crypto-shredding a tenant simply
        drops its rows from the snapshot and one bad row can't sink the whole load.
        """
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT key, model, tenant, vector FROM entries ORDER BY rowid DESC LIMIT ?",
                    (self._max,),
                ).fetchall()
        except Exception:
            return []  # fail open: a broken store degrades the index to in-memory-only
        entries = [
            entry
            for key, model, tenant, vector in rows
            if (entry := self._row_to_entry(key, model, tenant, bytes(vector))) is not None
        ]
        entries.reverse()  # DESC fetch -> return oldest first (chronological)
        return entries

    def _row_to_entry(
        self, key: str, model: str | None, tenant: str, blob: bytes
    ) -> SimilarityEntry | None:
        """Decode a row into an entry, or ``None`` to skip it (shredded/undecryptable/malformed)."""
        try:
            if self._provider is not None:
                cipher = self._provider.for_tenant(tenant)
                if cipher is None:
                    return None  # tenant shredded — key withheld, row inaccessible
                opened = cipher.open(key, blob)
                if opened is None:
                    return None  # wrong/rotated key or tamper — skip
                blob = opened
            return SimilarityEntry(vector=_decode_vector(blob), key=key, model=model, tenant=tenant)
        except Exception:
            return None  # malformed row -> skip, don't drop the whole snapshot

    def append(self, entry: SimilarityEntry) -> None:
        """Persist ``entry`` (sealing the vector when a provider is set); FIFO-evict; no-op on err.

        With a provider, a shredded tenant (``for_tenant`` → ``None``) is **not** persisted — as
        the encrypted exact cache skips writes for a withheld key.
        """
        try:
            blob = _encode_vector(entry.vector)
            if self._provider is not None:
                cipher = self._provider.for_tenant(entry.tenant)
                if cipher is None:
                    return  # tenant shredded — do not persist
                blob = cipher.seal(entry.key, blob)
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO entries (key, model, tenant, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (entry.key, entry.model, entry.tenant, blob, self._clock.now()),
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
