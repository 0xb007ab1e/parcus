"""Persistent metrics store: durable per-stage reduction/accuracy + eval scores.

A :class:`SqliteMetricsSink` records each request's per-stage breakdown to SQLite (so totals
survive restarts) and can fold in offline eval-gate scores. ``snapshot`` aggregates both into
the same shape the in-process :class:`~parcus.obs.sinks.AggregateSink` produces, plus an
``evals`` section. Recording is **best-effort** — it never raises into the request path. The DB
holds counts/metadata only (no prompt/response content) and is created ``0600``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any

from parcus.obs.events import SavingsEvent

__all__ = ["SqliteMetricsSink"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dialect       TEXT NOT NULL,
    cache         TEXT NOT NULL,
    tokens_before INTEGER NOT NULL,
    tokens_after  INTEGER NOT NULL,
    status_code   INTEGER NOT NULL,
    created_at    REAL NOT NULL,
    tenant        TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS stage_events (
    stage         TEXT NOT NULL,
    tokens_before INTEGER NOT NULL,
    tokens_after  INTEGER NOT NULL,
    ok            INTEGER
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    score      REAL NOT NULL,
    passed     INTEGER NOT NULL,
    created_at REAL NOT NULL
);
"""


class SqliteMetricsSink:
    """A persistent :class:`~parcus.obs.sinks.MetricsSink` over SQLite.

    Args:
        path: Database path; ``":memory:"`` for an ephemeral store.
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the metrics DB, set owner-only permissions, ensure the schema."""
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            try:
                os.chmod(path, 0o600)  # counts only, but treat as confidential
            except OSError:
                pass
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add the tenant column to a pre-existing events table (idempotent expand migration)."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        if "tenant" not in columns:
            self._conn.execute("ALTER TABLE events ADD COLUMN tenant TEXT NOT NULL DEFAULT ''")

    def record(self, event: SavingsEvent) -> None:
        """Persist a request's totals + per-stage breakdown (best-effort; never raises)."""
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO events "
                    "(dialect, cache, tokens_before, tokens_after, status_code, created_at, "
                    "tenant) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.dialect,
                        event.cache,
                        event.tokens_before,
                        event.tokens_after,
                        event.status_code,
                        time.time(),
                        event.tenant,
                    ),
                )
                _ = cursor.lastrowid
                self._conn.executemany(
                    "INSERT INTO stage_events (stage, tokens_before, tokens_after, ok) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (s.stage, s.tokens_before, s.tokens_after, _as_int(s.ok))
                        for s in event.stages
                    ],
                )
        except Exception:
            return

    def record_eval(self, kind: str, score: float, passed: bool) -> None:
        """Persist an offline eval-gate result (best-effort; never raises)."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO eval_runs (kind, score, passed, created_at) VALUES (?, ?, ?, ?)",
                    (kind, score, int(passed), time.time()),
                )
        except Exception:
            return

    def snapshot(self) -> dict[str, Any]:
        """Aggregate persisted metrics into per-stage reduction/accuracy + totals + eval scores."""
        with self._lock:
            requests, hits, before, after = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cache = 'hit'), 0), "
                "COALESCE(SUM(tokens_before), 0), COALESCE(SUM(tokens_after), 0) FROM events"
            ).fetchone()
            stage_rows = self._conn.execute(
                "SELECT stage, SUM(tokens_before), SUM(tokens_after), COUNT(ok), "
                "COALESCE(SUM(ok), 0) FROM stage_events GROUP BY stage"
            ).fetchall()
            eval_rows = self._conn.execute(
                "SELECT kind, score, passed FROM eval_runs e "
                "WHERE id = (SELECT MAX(id) FROM eval_runs WHERE kind = e.kind)"
            ).fetchall()
            # Per-tenant attribution (only credentialed tenants; '' = single-tenant noise).
            tenant_rows = self._conn.execute(
                "SELECT tenant, COUNT(*), COALESCE(SUM(tokens_before), 0), "
                "COALESCE(SUM(tokens_after), 0) FROM events WHERE tenant != '' GROUP BY tenant"
            ).fetchall()

        saved = before - after
        return {
            "requests": requests,
            "cache_hits": hits,
            "cache_hit_rate": round(hits / requests, 4) if requests else 0.0,
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": saved,
            "overall_ratio": round(saved / before, 4) if before else 0.0,
            "stages": {
                stage: _stage_summary(sb, sa, checked, oks)
                for stage, sb, sa, checked, oks in stage_rows
            },
            "evals": {
                kind: {"score": round(score, 4), "passed": bool(passed)}
                for kind, score, passed in eval_rows
            },
            "by_tenant": {
                tenant: _tenant_summary(reqs, tb, ta) for tenant, reqs, tb, ta in tenant_rows
            },
        }

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Close the connection on GC — a backstop; deterministic cleanup is ``close()``."""
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()


def _as_int(ok: bool | None) -> int | None:
    return None if ok is None else int(ok)


def _tenant_summary(requests: int, before: int, after: int) -> dict[str, Any]:
    saved = before - after
    return {
        "requests": requests,
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": saved,
        "reduction": round(saved / before, 4) if before else 0.0,
    }


def _stage_summary(before: int, after: int, checked: int, oks: int) -> dict[str, Any]:
    saved = before - after
    return {
        "tokens_before": before,
        "tokens_after": after,
        "tokens_saved": saved,
        "reduction": round(saved / before, 4) if before else 0.0,
        "accuracy": round(oks / checked, 4) if checked else None,
        "checked": checked,
    }
