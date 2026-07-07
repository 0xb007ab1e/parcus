"""Unit tests for the context-cache registrars (ADR 0010).

Covers the ``ContextCacheHandle`` value type, the fail-open ``NullContextCacheRegistrar`` default,
and the ``GeminiContextCacheRegistrar`` lifecycle policy (reuse, expiry-as-miss, spend cap,
credential/model scoping, fail-open on I/O error, local pruning) driven through an **injected
async** ``create`` fake so no provider/network is touched. The registrar is async and
credential-scoped (ADR 0010 Updates 2026-07-04 / 2026-07-07); the real ``google-genai`` wiring in
``gemini_registrar`` is ``# pragma: no cover`` (needs the extra + a live key).
"""

from __future__ import annotations

from parcus.cache.context_cache import (
    GeminiContextCacheRegistrar,
    NullContextCacheRegistrar,
)
from parcus.model import ContextCacheHandle
from parcus.ports import ContextCacheRegistrar

KEY_A = "AIza-caller-a"
KEY_B = "AIza-caller-b"


class FakeClock:
    """A manually-advanced clock implementing ClockPort, for deterministic TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class RecordingCreator:
    """Fake async create I/O: records (prefix, model, credential) and hands out unique names."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str]] = []
        self._n = 0

    async def create(self, prefix: str, model: str | None, credential: str) -> str:
        self.calls.append((prefix, model, credential))
        self._n += 1
        return f"cachedContents/{self._n}"


async def _raise_create(prefix: str, model: str | None, credential: str) -> str:
    raise RuntimeError("provider 500")


async def _const_create(prefix: str, model: str | None, credential: str) -> str:
    return "cachedContents/x"


def _gemini(
    clock: FakeClock,
    creator: RecordingCreator,
    *,
    ttl_seconds: int = 100,
    max_entries: int = 64,
) -> GeminiContextCacheRegistrar:
    return GeminiContextCacheRegistrar(
        clock=clock,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        create=creator.create,
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
        assert (
            await reg.ensure("a long stable prefix", model="gemini-2.5-flash", credential=KEY_A)
            is None
        )
        assert await reg.ensure("x", model=None, credential=KEY_B) is None

    async def test_evict_is_noop(self) -> None:
        assert await NullContextCacheRegistrar().evict_expired() is None


# --- GeminiContextCacheRegistrar lifecycle ----------------------------------------------------


class TestGeminiRegistrarCreateAndReuse:
    async def test_creates_on_first_miss(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator)
        handle = await reg.ensure("PREFIX", model="gemini-2.5-flash", credential=KEY_A)
        assert handle is not None
        assert handle.name == "cachedContents/1"
        assert handle.model == "gemini-2.5-flash"
        assert handle.expires_at == 1100.0  # now(1000) + ttl(100)
        assert creator.calls == [("PREFIX", "gemini-2.5-flash", KEY_A)]  # created under caller key

    async def test_reuses_live_handle_without_recreating(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator)
        first = await reg.ensure("PREFIX", model="m", credential=KEY_A)
        clock.advance(50)  # still within the 100s TTL
        second = await reg.ensure("PREFIX", model="m", credential=KEY_A)
        assert second == first
        assert len(creator.calls) == 1  # no second create

    async def test_expired_handle_is_recreated(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator)
        await reg.ensure("PREFIX", model="m", credential=KEY_A)
        clock.advance(150)  # past the 100s TTL
        again = await reg.ensure("PREFIX", model="m", credential=KEY_A)
        assert again is not None
        assert again.name == "cachedContents/2"
        assert len(creator.calls) == 2


class TestGeminiRegistrarScoping:
    async def test_distinct_credentials_get_distinct_handles(self) -> None:
        # The core isolation guarantee: two callers' keys never share a handle (even same prefix).
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator)
        a = await reg.ensure("PREFIX", model="m", credential=KEY_A)
        b = await reg.ensure("PREFIX", model="m", credential=KEY_B)
        assert a is not None and b is not None and a.name != b.name

    async def test_distinct_models_get_distinct_handles(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator)
        a = await reg.ensure("PREFIX", model="m1", credential=KEY_A)
        b = await reg.ensure("PREFIX", model="m2", credential=KEY_A)
        assert a is not None and b is not None and a.name != b.name

    async def test_none_model_is_handled(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator)
        handle = await reg.ensure("PREFIX", model=None, credential=KEY_A)
        assert handle is not None
        assert handle.model == ""  # normalised


class TestGeminiRegistrarSpendCap:
    async def test_cap_blocks_new_creation_but_serves_existing(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator, max_entries=1)
        first = await reg.ensure("PREFIX-A", model="m", credential=KEY_A)
        # A different prefix would be a 2nd live cache — capped → fail open (inline send).
        assert await reg.ensure("PREFIX-B", model="m", credential=KEY_A) is None
        # The already-registered prefix is still served from the held handle.
        assert await reg.ensure("PREFIX-A", model="m", credential=KEY_A) == first
        assert len(creator.calls) == 1

    async def test_cap_frees_after_expiry(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator, max_entries=1)
        await reg.ensure("PREFIX-A", model="m", credential=KEY_A)
        clock.advance(150)  # A expires → no longer counts against the cap
        assert await reg.ensure("PREFIX-B", model="m", credential=KEY_A) is not None


class TestGeminiRegistrarFailOpen:
    async def test_create_error_returns_none(self) -> None:
        reg = GeminiContextCacheRegistrar(
            clock=FakeClock(), ttl_seconds=100, max_entries=64, create=_raise_create
        )
        assert await reg.ensure("PREFIX", model="m", credential=KEY_A) is None  # never raises


class TestGeminiRegistrarPruning:
    async def test_evict_drops_expired_and_frees_capacity(self) -> None:
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator, ttl_seconds=100, max_entries=2)
        await reg.ensure("OLD", model="m", credential=KEY_A)  # expires at 1100
        clock.advance(150)  # now 1150 — OLD expired
        await reg.evict_expired()  # prunes OLD locally (no remote call)
        # A fresh prefix now creates cleanly (map no longer holds the dead OLD entry).
        assert await reg.ensure("NEW", model="m", credential=KEY_A) is not None
        assert len(creator.calls) == 2

    async def test_ensure_opportunistically_prunes_expired(self) -> None:
        # With no evict call, ensure() itself keeps the map bounded once len exceeds cap*factor
        # (=4 here): the prune runs at the top of the ensure *after* the map has grown past it.
        clock, creator = FakeClock(), RecordingCreator()
        reg = _gemini(clock, creator, ttl_seconds=100, max_entries=1)
        for i in range(6):
            await reg.ensure(f"P{i}", model="m", credential=KEY_A)
            clock.advance(200)  # each entry expires before the next ensure
        # The 6th ensure saw len=5 (>4) and pruned all the expired entries before creating.
        assert len(reg._handles) <= 2  # bounded, not 6


# --- Structural conformance to the port -------------------------------------------------------


def test_registrars_satisfy_the_port() -> None:
    assert isinstance(NullContextCacheRegistrar(), ContextCacheRegistrar)
    reg = GeminiContextCacheRegistrar(
        clock=FakeClock(), ttl_seconds=100, max_entries=64, create=_const_create
    )
    assert isinstance(reg, ContextCacheRegistrar)
