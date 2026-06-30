"""SQLite-backed exact-match response cache.

Stores responses verbatim for byte-for-byte replay, keyed by the one-way hash from
:func:`parcus.cache.key.compute_key` (prompts themselves are never stored). The store is
**confidential** (see the threat model): the backing file is created ``0600`` and entries
carry a TTL with lazy expiry.

Every operation **fails open**: a get/put error returns ``None``/no-ops rather than raising,
because the cache is a performance layer and the system must be correct when it is empty or
unavailable.
"""

from __future__ import annotations

import os
import sqlite3
import threading

from parcus.cache.clock import SystemClock
from parcus.model import CachedResponse
from parcus.ports import ClockPort

__all__ = ["SqliteCache"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key          TEXT PRIMARY KEY,
    status_code  INTEGER NOT NULL,
    content_type TEXT,
    body         BLOB NOT NULL,
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL
)
"""


class SqliteCache:
    """A confidential, TTL-bound response cache over SQLite.

    Implements :class:`parcus.ports.CachePort`.

    Args:
        path: Database path; ``":memory:"`` for an ephemeral in-process cache.
        clock: Injected time source (defaults to :class:`SystemClock`) for TTL/testability.
    """

    def __init__(self, path: str = ":memory:", clock: ClockPort | None = None) -> None:
        """Open (or create) the cache database and ensure its schema and permissions."""
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

    def get(self, key: str, *, tenant: str = "") -> CachedResponse | None:
        """Return the unexpired cached response for ``key``, else ``None`` (fails open).

        ``tenant`` is accepted for interface parity (the encrypting wrapper uses it) and ignored
        here — entries are already keyed by the tenant-namespaced hash.
        """
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT status_code, content_type, body, expires_at "
                    "FROM responses WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                status_code, content_type, body, expires_at = row
                if expires_at <= self._clock.now():
                    self._conn.execute("DELETE FROM responses WHERE key = ?", (key,))
                    self._conn.commit()
                    return None
                return CachedResponse(
                    status_code=int(status_code),
                    body=bytes(body),
                    content_type=content_type,
                )
        except Exception:
            # Fail open: a cache read must never break the request path.
            return None

    def put(self, key: str, value: CachedResponse, ttl_seconds: int, *, tenant: str = "") -> None:
        """Store ``value`` under ``key`` for ``ttl_seconds`` (no-op on error or ttl<=0).

        ``tenant`` is accepted for interface parity and ignored here.
        """
        if ttl_seconds <= 0:
            return
        try:
            now = self._clock.now()
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO responses "
                    "(key, status_code, content_type, body, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        value.status_code,
                        value.content_type,
                        value.body,
                        now,
                        now + ttl_seconds,
                    ),
                )
        except Exception:
            # Fail open: a cache write must never break the request path.
            return

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Close the connection on GC — a backstop; deterministic cleanup is ``close()``.

        Guards a leaked connection (and its noisy ResourceWarning) if an instance is GC'd
        without an explicit close, e.g. a short-lived store created inline in a test.
        """
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
