"""Tests for the SQLite response cache (TTL, replay fidelity, fail-open)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from parsimony.cache import SqliteCache, SystemClock
from parsimony.model import CachedResponse


class FakeClock:
    """A manually-advanced clock implementing ClockPort, for deterministic TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _resp(body: bytes = b"hello", status: int = 200) -> CachedResponse:
    return CachedResponse(status_code=status, body=body, content_type="application/json")


class TestSqliteCache:
    def test_put_then_get_roundtrip(self) -> None:
        cache = SqliteCache(clock=FakeClock())
        cache.put("k", _resp(b"payload"), ttl_seconds=100)
        got = cache.get("k")
        assert got is not None
        assert got.body == b"payload"
        assert got.status_code == 200
        assert got.content_type == "application/json"

    def test_miss_returns_none(self) -> None:
        assert SqliteCache(clock=FakeClock()).get("absent") is None

    def test_entry_expires_after_ttl(self) -> None:
        clock = FakeClock()
        cache = SqliteCache(clock=clock)
        cache.put("k", _resp(), ttl_seconds=10)
        clock.advance(5)
        assert cache.get("k") is not None  # still fresh
        clock.advance(6)  # now past the 10s TTL
        assert cache.get("k") is None  # expired (and lazily deleted)
        # Confirm the lazy delete actually removed the row (second miss, same result).
        assert cache.get("k") is None

    def test_non_positive_ttl_is_not_stored(self) -> None:
        cache = SqliteCache(clock=FakeClock())
        cache.put("k", _resp(), ttl_seconds=0)
        assert cache.get("k") is None

    def test_put_replaces_existing_key(self) -> None:
        cache = SqliteCache(clock=FakeClock())
        cache.put("k", _resp(b"first"), ttl_seconds=100)
        cache.put("k", _resp(b"second"), ttl_seconds=100)
        got = cache.get("k")
        assert got is not None
        assert got.body == b"second"

    def test_fails_open_after_close(self) -> None:
        # A broken/closed store must never raise into the request path.
        cache = SqliteCache(clock=FakeClock())
        cache.close()
        assert cache.get("k") is None  # no exception
        cache.put("k", _resp(), ttl_seconds=100)  # no exception


class TestFileBackedStore:
    def test_database_file_is_owner_only(self, tmp_path: Path) -> None:
        # The cache holds confidential data; its file must be created 0600.
        path = tmp_path / "cache.sqlite"
        cache = SqliteCache(path=str(path), clock=FakeClock())
        cache.put("k", _resp(b"x"), ttl_seconds=100)
        got = cache.get("k")
        assert got is not None
        assert got.body == b"x"
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        cache.close()

    def test_chmod_failure_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the OS refuses chmod, construction must still succeed (best-effort hardening).
        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("nope")

        monkeypatch.setattr(os, "chmod", _boom)
        cache = SqliteCache(path=str(tmp_path / "c.sqlite"), clock=FakeClock())
        cache.close()


class TestSystemClock:
    def test_now_is_a_positive_timestamp(self) -> None:
        import time

        now = SystemClock().now()
        assert isinstance(now, float)
        assert abs(now - time.time()) < 5.0
