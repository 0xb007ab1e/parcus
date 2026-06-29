"""Concurrency / race tests for the shared-mutable-state components.

The rate limiter (token bucket), the cache, and per-tenant scoping are touched concurrently —
the rate limiter and cache key on the async event loop, the SQLite cache also from worker
threads. These are the classic spots for over-admission and TOCTOU (`topic-concurrency`). The
tests assert the invariants hold under concurrent access:

* the limiter admits **exactly capacity**, never more, under a burst of concurrent checks
  (its ``check`` has no ``await`` inside, so the single-threaded loop can't interleave it — this
  is the regression guard if that ever changes);
* per-tenant buckets stay independent under interleaved load;
* the SQLite cache is consistent and error-free under concurrent threads (it guards with a lock);
* multi-tenant cache scoping never leaks across tenants even under concurrent interleaving (E1).
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from parcus.cache import CachePolicy, SqliteCache
from parcus.compress import LosslessCompressor
from parcus.model import CachedResponse
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.quota import RateLimit, RateLimiter
from parcus.redact import Redactor


class _FrozenClock:
    """A fixed monotonic source (no refill happens), so admission count is deterministic."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _FakeUpstream:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.calls += 1
        await asyncio.sleep(0)  # a real yield point — lets other requests interleave
        return UpstreamResponse(200, (("content-type", "application/json"),), b'{"ok":1}')


# --- rate limiter -----------------------------------------------------------------------------


class TestRateLimiterConcurrency:
    async def test_admits_exactly_capacity_under_burst(self) -> None:
        limiter = RateLimiter(RateLimit(capacity=5, refill_per_sec=0.0), time_source=_FrozenClock())

        async def one() -> bool:
            return limiter.check("t").allowed

        results = await asyncio.gather(*[one() for _ in range(50)])
        assert sum(results) == 5  # never over-admits, no matter the concurrency

    async def test_per_tenant_buckets_independent_under_interleave(self) -> None:
        limiter = RateLimiter(RateLimit(capacity=3, refill_per_sec=0.0), time_source=_FrozenClock())

        async def one(tenant: str) -> tuple[str, bool]:
            return tenant, limiter.check(tenant).allowed

        calls = [one("a") for _ in range(20)] + [one("b") for _ in range(20)]
        random.Random(0).shuffle(calls)  # noqa: S311 — deterministic interleave, not crypto
        results = await asyncio.gather(*calls)
        assert sum(1 for t, ok in results if t == "a" and ok) == 3
        assert sum(1 for t, ok in results if t == "b" and ok) == 3


# --- SQLite cache (thread safety) -------------------------------------------------------------


class TestCacheConcurrency:
    async def test_concurrent_put_get_round_trips(self, tmp_path: Path) -> None:
        cache = SqliteCache(str(tmp_path / "c.db"))

        def work(i: int) -> bool:
            key = f"k{i}"
            cache.put(key, CachedResponse(200, f"v{i}".encode(), "application/json"), 3600)
            got = cache.get(key)
            return got is not None and got.body == f"v{i}".encode()

        # asyncio.to_thread runs these on real worker threads -> exercises the cache's lock.
        results = await asyncio.gather(*[asyncio.to_thread(work, i) for i in range(100)])
        assert all(results)  # every put/get round-tripped; no corruption or threading error

    async def test_concurrent_writers_same_key_leave_a_valid_value(self, tmp_path: Path) -> None:
        cache = SqliteCache(str(tmp_path / "c.db"))
        bodies = {f"v{i}".encode() for i in range(50)}

        def write(i: int) -> None:
            cache.put("shared", CachedResponse(200, f"v{i}".encode(), "application/json"), 3600)

        await asyncio.gather(*[asyncio.to_thread(write, i) for i in range(50)])
        final = cache.get("shared")
        assert final is not None and final.body in bodies  # a consistent winner, not corruption


# --- engine integration (rate limit + multi-tenant cache under concurrent load) ---------------


def _engine(upstream: _FakeUpstream, **kw: object) -> ProxyEngine:
    return ProxyEngine(
        upstream=upstream,
        compressor=LosslessCompressor(),
        cache=kw.get("cache", SqliteCache()),  # type: ignore[arg-type]
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            cache_enabled=bool(kw.get("cache_enabled", True)),
            multi_tenant=bool(kw.get("multi_tenant", False)),
        ),
        rate_limiter=kw.get("rate_limiter"),  # type: ignore[arg-type]
    )


def _body(text: str) -> bytes:
    return json.dumps({"model": "m", "messages": [{"role": "user", "content": text}]}).encode()


class TestEngineConcurrency:
    async def test_rate_limit_admits_exactly_capacity_through_the_engine(self) -> None:
        up = _FakeUpstream()
        limiter = RateLimiter(RateLimit(capacity=3, refill_per_sec=0.0), time_source=_FrozenClock())
        eng = _engine(up, cache_enabled=False, rate_limiter=limiter)
        headers = [("x-api-key", "k")]
        results = await asyncio.gather(
            *[eng.handle("POST", "/v1/messages", headers, _body(f"q{i}")) for i in range(30)]
        )
        statuses = [r.status_code for r in results]
        assert statuses.count(200) == 3  # exactly capacity reached upstream
        assert statuses.count(429) == 27  # the rest shed, before any upstream call
        assert up.calls == 3

    async def test_no_cross_tenant_cache_leak_under_concurrent_load(self) -> None:
        # Same body, two tenants, fired concurrently and interleaved: neither may serve the
        # other's cached response (BOLA / threat E1) even under racing.
        up = _FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=True)
        body = _body("identical across tenants")
        calls = []
        for _ in range(10):
            calls.append(eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body))
            calls.append(eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-b")], body))
        random.Random(1).shuffle(calls)  # noqa: S311 — deterministic interleave, not crypto
        results = await asyncio.gather(*calls)
        assert all(r.status_code == 200 for r in results)
        # Each tenant's first request misses and reaches upstream; cross-tenant hits would have
        # let one tenant serve the other -> upstream called at least once per tenant.
        assert up.calls >= 2
