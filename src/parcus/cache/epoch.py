"""Per-tenant key epoch — a monotonic counter that drives irreversible crypto-shredding.

Crypto-shredding via a withheld-key *set* (ADR 0007) has a gap: removing a tenant from the set
(or restarting without it) makes its still-unexpired ciphertext readable again. This module fixes
that with a **key epoch** per tenant: the epoch is folded into the DEK derivation
(:class:`~parcus.cache.encryption.EpochCipherProvider`), and to shred a tenant you ``bump`` its
epoch. The epoch is **monotonic** (only ever increases) and, in the SQLite store, **persistent**
(survives restart) — so a shred has no undo path: the provider derives the new epoch's DEK and
never again the old one, leaving the pre-bump ciphertext permanently inaccessible (it ages out by
TTL). See ADR 0009.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Protocol, runtime_checkable

__all__ = ["EpochStore", "InMemoryEpochStore", "SqliteEpochStore"]

_SCHEMA = "CREATE TABLE IF NOT EXISTS key_epochs (tenant TEXT PRIMARY KEY, epoch INTEGER NOT NULL)"


@runtime_checkable
class EpochStore(Protocol):
    """Tracks a per-tenant key epoch that only ever increases."""

    def epoch(self, tenant: str) -> int:
        """Return the tenant's current epoch (``0`` if it has never been bumped)."""
        ...

    def bump(self, tenant: str) -> int:
        """Increase the tenant's epoch by one and return the new value (the shred operation)."""
        ...


class InMemoryEpochStore:
    """A non-persistent :class:`EpochStore` (tests / ephemeral use; epochs reset on restart)."""

    def __init__(self) -> None:
        """Start with all tenants at epoch 0."""
        self._epochs: dict[str, int] = {}
        self._lock = threading.Lock()

    def epoch(self, tenant: str) -> int:
        """Return the tenant's current epoch (0 by default)."""
        with self._lock:
            return self._epochs.get(tenant, 0)

    def bump(self, tenant: str) -> int:
        """Increment the tenant's epoch and return it."""
        with self._lock:
            new = self._epochs.get(tenant, 0) + 1
            self._epochs[tenant] = new
            return new


class SqliteEpochStore:
    """A persistent :class:`EpochStore` in SQLite (``0600``).

    Epochs survive process restart and can only increase (the bump is a monotonic ``+1`` in the
    database), so a shred is durable and has no undo — the key property that makes crypto-shredding
    irreversible. Thread-safe via a lock + ``check_same_thread=False``.

    Args:
        path: Database path; ``":memory:"`` for an ephemeral store.
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open (and 0600-protect) the database and ensure the schema exists."""
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            os.chmod(path, 0o600)  # epoch state is security-relevant — owner-only
        with self._conn:
            self._conn.execute(_SCHEMA)

    def epoch(self, tenant: str) -> int:
        """Return the tenant's persisted epoch (0 if absent)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT epoch FROM key_epochs WHERE tenant = ?", (tenant,)
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def bump(self, tenant: str) -> int:
        """Atomically increment (or initialise to 1) the tenant's epoch and return it."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO key_epochs (tenant, epoch) VALUES (?, 1) "
                "ON CONFLICT(tenant) DO UPDATE SET epoch = epoch + 1",
                (tenant,),
            )
            row = self._conn.execute(
                "SELECT epoch FROM key_epochs WHERE tenant = ?", (tenant,)
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
