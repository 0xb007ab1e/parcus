"""Unit tests for the context-cache registrars (ADR 0010).

Covers the ``ContextCacheHandle`` value type, the fail-open ``NullContextCacheRegistrar`` default,
and the ``GeminiContextCacheRegistrar`` lifecycle policy (reuse, expiry-as-miss, spend cap,
fail-open on I/O error, TTL eviction) driven through **injected async** create/delete fakes so no
provider/network is touched. The registrar port is async (ADR 0010 Update 2026-07-04); the real
``google-genai`` wiring in ``gemini_registrar`` is ``# pragma: no cover`` (needs the extra + a
live key).
"""

from __future__ import annotations

from parcus.cache.context_cache import (
    GeminiContextCacheRegistrar,
    NullContextCacheRegistrar,
)
from parcus.model import ContextCacheHandle
from parcus.ports import ContextCacheRegistrar


class FakeClock:
    """A manually-advanced clock implementing ClockPort, for deterministic TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class RecordingProvider:
    """Fake async create/delete I/O: hands out deterministic names and records delete calls."""

    def __init__(self) -> None:
        self.created: list[tuple[str, str | None]] = []
        self.deleted: list[str] = []
        self._n = 0

    async def create(self, prefix: str, model: str | None) -> str:
        self.created.append((prefix, model))
        self._n += 1
        return f"cachedContents/{self._n}"

    async def delete(self, name: str) -> None:
        self.deleted.append(name)


async def _raise_create(prefix: str, model: str | None) -> str:
    raise RuntimeError("provider 500")


async def _raise_delete(name: str) -> None:
    raise RuntimeError("delete failed")


async def _const_create(prefix: str, model: str | None) -> str:
    return "cachedContents/1"


async def _noop_delete(name: str) -> None:
    return None


def _gemini(
    clock: FakeClock,
    provider: RecordingProvider,
    *,
    ttl_seconds: int = 100,
    max_entries: int = 64,
) -> GeminiContextCacheRegistrar:
    return GeminiContextCacheRegistrar(
        clock=clock,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        create=provider.create,
        delete=provider.delete,
    )


# --- ContextCacheHandle -----------------------------------------------------------------------


class TestContextCacheHandle:
    def test_fields(self) -> None:
        h = ContextCacheHandle(name="cachedContents/x", model="gemini-2.5-flash", expires_at=1234.0)
        assert h.name == "cachedContents/x"
        assert h.model == "gemini-2.5-flash"
        assert h.expires_at == 1234.0

    def test_is_expired_boundaries(self) -> None:
        h = ContextCacheHandle(name="n", model="m", expires_at=100.0)
        assert h.is_expired(100.0) is True  # at expiry: expired (>=)
        assert h.is_expired(101.0) is True
        assert h.is_expired(99.9) is False


# --- NullContextCacheRegistrar ----------------------------------------------------------------


class TestNullContextCacheRegistrar:
    async def test_ensure_always_none(self) -> None:
        reg = NullContextCacheRegistrar()
        assert await reg.ensure("a long stable prefix", model="gemini-2.5-flash") is None
        assert await reg.ensure("x", model=None, tenant="t1") is None

    async def test_evict_is_noop(self) -> None:
        assert await NullContextCacheRegistrar().evict_expired() is None


# --- GeminiContextCacheRegistrar lifecycle ----------------------------------------------------


class TestGeminiRegistrarCreateAndReuse:
    async def test_creates_on_first_miss(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider)
        handle = await reg.ensure("PREFIX", model="gemini-2.5-flash", tenant="t1")
        assert handle is not None
        assert handle.name == "cachedContents/1"
        assert handle.model == "gemini-2.5-flash"
        assert handle.expires_at == 1100.0  # now(1000) + ttl(100)
        assert provider.created == [("PREFIX", "gemini-2.5-flash")]

    async def test_reuses_live_handle_without_recreating(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider)
        first = await reg.ensure("PREFIX", model="m", tenant="t1")
        clock.advance(50)  # still within the 100s TTL
        second = await reg.ensure("PREFIX", model="m", tenant="t1")
        assert second == first
        assert len(provider.created) == 1  # no second create

    async def test_expired_handle_is_recreated(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider)
        await reg.ensure("PREFIX", model="m", tenant="t1")
        clock.advance(150)  # past the 100s TTL
        again = await reg.ensure("PREFIX", model="m", tenant="t1")
        assert again is not None
        assert again.name == "cachedContents/2"
        assert len(provider.created) == 2


class TestGeminiRegistrarScoping:
    async def test_distinct_tenants_get_distinct_handles(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider)
        a = await reg.ensure("PREFIX", model="m", tenant="t1")
        b = await reg.ensure("PREFIX", model="m", tenant="t2")
        assert a is not None and b is not None and a.name != b.name  # never shared across tenants

    async def test_distinct_models_get_distinct_handles(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider)
        a = await reg.ensure("PREFIX", model="m1", tenant="t1")
        b = await reg.ensure("PREFIX", model="m2", tenant="t1")
        assert a is not None and b is not None and a.name != b.name

    async def test_none_model_is_handled(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider)
        handle = await reg.ensure("PREFIX", model=None, tenant="t1")
        assert handle is not None
        assert handle.model == ""  # normalised


class TestGeminiRegistrarSpendCap:
    async def test_cap_blocks_new_creation_but_serves_existing(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider, max_entries=1)
        first = await reg.ensure("PREFIX-A", model="m", tenant="t1")
        # A different prefix would be a 2nd live cache — capped → fail open (inline send).
        assert await reg.ensure("PREFIX-B", model="m", tenant="t1") is None
        # The already-registered prefix is still served from the held handle.
        assert await reg.ensure("PREFIX-A", model="m", tenant="t1") == first
        assert len(provider.created) == 1

    async def test_cap_frees_after_expiry(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider, max_entries=1)
        await reg.ensure("PREFIX-A", model="m", tenant="t1")
        clock.advance(150)  # A expires → no longer counts against the cap
        assert await reg.ensure("PREFIX-B", model="m", tenant="t1") is not None


class TestGeminiRegistrarFailOpen:
    async def test_create_error_returns_none(self) -> None:
        reg = GeminiContextCacheRegistrar(
            clock=FakeClock(),
            ttl_seconds=100,
            max_entries=64,
            create=_raise_create,
            delete=_noop_delete,
        )
        assert await reg.ensure("PREFIX", model="m", tenant="t1") is None  # fail open, never raises


class TestGeminiRegistrarEviction:
    async def test_evict_deletes_only_expired(self) -> None:
        clock, provider = FakeClock(), RecordingProvider()
        reg = _gemini(clock, provider, ttl_seconds=100)
        await reg.ensure("OLD", model="m", tenant="t1")  # expires at 1100
        clock.advance(60)  # now 1060
        await reg.ensure("NEW", model="m", tenant="t1")  # expires at 1160
        clock.advance(60)  # now 1120 — OLD expired, NEW still live
        await reg.evict_expired()
        assert provider.deleted == ["cachedContents/1"]  # only OLD deleted
        # NEW is still served without a re-create.
        assert await reg.ensure("NEW", model="m", tenant="t1") is not None
        assert len(provider.created) == 2

    async def test_evict_swallows_delete_errors(self) -> None:
        clock = FakeClock()
        reg = GeminiContextCacheRegistrar(
            clock=clock,
            ttl_seconds=100,
            max_entries=64,
            create=_const_create,
            delete=_raise_delete,
        )
        await reg.ensure("OLD", model="m", tenant="t1")
        clock.advance(150)
        await reg.evict_expired()  # must not raise despite the delete error


# --- Structural conformance to the port -------------------------------------------------------


def test_registrars_satisfy_the_port() -> None:
    assert isinstance(NullContextCacheRegistrar(), ContextCacheRegistrar)
    reg = GeminiContextCacheRegistrar(
        clock=FakeClock(),
        ttl_seconds=100,
        max_entries=64,
        create=_const_create,
        delete=_noop_delete,
    )
    assert isinstance(reg, ContextCacheRegistrar)
