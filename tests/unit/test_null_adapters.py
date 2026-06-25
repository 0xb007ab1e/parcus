"""Tests for the no-op compressor and cache used when those features are disabled."""

from __future__ import annotations

from parcus.cache import NullCache
from parcus.compress import NullCompressor
from parcus.model import CachedResponse, CanonicalRequest, Dialect


def test_null_compressor_returns_request_unchanged() -> None:
    req = CanonicalRequest(dialect=Dialect.ANTHROPIC, model="m", messages=())
    out, stats = NullCompressor().compress(req)
    assert out is req
    assert stats == ()


def test_null_cache_always_misses() -> None:
    cache = NullCache()
    assert cache.get("k") is None
    cache.put("k", CachedResponse(status_code=200, body=b"x"), 100)  # no-op
    assert cache.get("k") is None
