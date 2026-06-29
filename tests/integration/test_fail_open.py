"""Fault-injection tests for the fail-open guarantee — parcus's prime directive.

parcus must be *invisible*: "on any uncertainty, forward the original request and serve the
real response." The ports & adapters design makes that directly testable — we inject an adapter
that **raises** at each seam and assert the request still succeeds with the genuine upstream
response, never a 5xx and never an exception out of the engine.

Two kinds of fault are covered:

* **Adapter contract violations** (a tokenizer / redactor / cache / similarity index / memory
  that throws): the engine degrades — it skips that optimization and forwards anyway (defense
  in depth; master §2). Optimization fails *open*.
* **Security decisions still fail *closed***: a detected secret means the request is forwarded
  (open) but **not cached** (closed). Availability never overrides confidentiality.

Plus the input-side fail-open paths: unparseable / non-canonical bodies pass through verbatim.
"""

from __future__ import annotations

import json

from parcus.cache import CachePolicy, SqliteCache
from parcus.compress import LosslessCompressor
from parcus.model import CanonicalRequest, CompressionStats
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor

REAL = UpstreamResponse(200, (("content-type", "application/json"),), b'{"answer":42}')


class FakeUpstream:
    """Records the forwarded request and returns a fixed, recognisable response."""

    def __init__(self) -> None:
        self.calls = 0
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.calls += 1
        self.last = request
        return REAL


# --- Adapters that violate their contract by raising at a specific seam -----------------------


class BoomCompressor:
    """compress() always raises."""

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        raise RuntimeError("compress boom")


class BoomTokenizer:
    """count() always raises (token metrics must never break the request)."""

    def count(self, text: str, model: str | None = None) -> int:
        raise RuntimeError("tokenizer boom")


class BoomRedactor:
    """The secret check raises (must fail closed for caching, open for forwarding)."""

    def redact(self, text: str) -> tuple[str, object]:
        raise RuntimeError("redact boom")

    def has_secret(self, text: str) -> bool:
        raise RuntimeError("has_secret boom")


class BoomGetCache(SqliteCache):
    """A cache whose reads raise."""

    def get(self, key: str, *, tenant: str = "") -> object:
        raise RuntimeError("cache get boom")


class BoomPutCache(SqliteCache):
    """A cache whose writes raise."""

    def put(self, key: str, value: object, ttl_seconds: int, *, tenant: str = "") -> None:
        raise RuntimeError("cache put boom")


class BoomLookupSimilarity:
    """A similarity index whose lookup raises."""

    def lookup(self, *, text: str, model: str | None, tenant: str) -> str | None:
        raise RuntimeError("similarity lookup boom")

    def remember(self, **kwargs: object) -> None:
        raise RuntimeError("similarity remember boom")


class BoomRememberSimilarity:
    """A similarity index that finds nothing but raises while remembering the new entry."""

    def lookup(self, *, text: str, model: str | None, tenant: str) -> str | None:
        return None

    def remember(self, **kwargs: object) -> None:
        raise RuntimeError("similarity remember boom")


class BoomMemory:
    """A memory adapter whose ingest raises."""

    def ingest(self, request: CanonicalRequest) -> None:
        raise RuntimeError("memory ingest boom")

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        return ()


def _engine(up: FakeUpstream, **kw: object) -> ProxyEngine:
    """Build an engine with the shipped defaults, overriding only the injected fault adapter."""
    return ProxyEngine(
        upstream=up,
        compressor=kw.get("compressor", LosslessCompressor()),  # type: ignore[arg-type]
        cache=kw.get("cache", SqliteCache()),  # type: ignore[arg-type]
        redactor=kw.get("redactor", Redactor()),  # type: ignore[arg-type]
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            memory_enabled=bool(kw.get("memory_enabled", False)),
        ),
        metrics=kw.get("metrics"),  # type: ignore[arg-type]
        memory=kw.get("memory"),  # type: ignore[arg-type]
        similarity=kw.get("similarity"),  # type: ignore[arg-type]
        tokenizer=kw.get("tokenizer"),  # type: ignore[arg-type]
    )


def _anthropic(content: str) -> bytes:
    return json.dumps(
        {"model": "claude-x", "messages": [{"role": "user", "content": content}]}
    ).encode()


async def _expect_served(eng: ProxyEngine, up: FakeUpstream, body: bytes) -> None:
    """Every fault case must end the same way: 200, one upstream call, the real response."""
    result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
    assert result.status_code == 200
    assert up.calls == 1  # the request reached the provider exactly once
    assert result.content == REAL.content  # the genuine upstream response was returned


class TestSeamFaultsFailOpen:
    """A raising adapter at any seam degrades to "forward + serve" — never a crash."""

    async def test_compressor_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, compressor=BoomCompressor())
        body = _anthropic("please   fix   this   \n\n\n\n")
        await _expect_served(eng, up, body)
        assert up.last is not None
        assert up.last.content == body  # the ORIGINAL body is forwarded unchanged

    async def test_tokenizer_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, tokenizer=BoomTokenizer())
        await _expect_served(eng, up, _anthropic("hello world"))

    async def test_redactor_raises_forwards_but_does_not_cache(self) -> None:
        # has_secret raising must fail closed for caching (no key) yet open for forwarding.
        up = FakeUpstream()
        eng = _engine(up, redactor=BoomRedactor(), cache=SqliteCache())
        body = _anthropic("compute something")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.calls == 2  # never cached -> both reached upstream

    async def test_cache_get_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=BoomGetCache())
        await _expect_served(eng, up, _anthropic("hi"))

    async def test_cache_put_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=BoomPutCache())
        await _expect_served(eng, up, _anthropic("hi"))

    async def test_similarity_lookup_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), similarity=BoomLookupSimilarity())
        await _expect_served(eng, up, _anthropic("hi"))

    async def test_similarity_remember_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), similarity=BoomRememberSimilarity())
        await _expect_served(eng, up, _anthropic("hi"))

    async def test_memory_ingest_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, memory=BoomMemory(), memory_enabled=True)
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("hi"))
        assert result.status_code == 200
        assert up.calls == 1
        assert result.meta["memory"] == "off"  # ingest failure degrades to no-memory


class TestMetricsStillEmittedUnderFault:
    """Observability must survive a fault too — exactly one event is still recorded."""

    async def test_one_event_even_when_a_seam_raises(self) -> None:
        class SpySink:
            def __init__(self) -> None:
                self.events: list[object] = []

            def record(self, event: object) -> None:
                self.events.append(event)

        spy = SpySink()
        up = FakeUpstream()
        eng = _engine(up, tokenizer=BoomTokenizer(), metrics=spy)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("hi"))
        assert len(spy.events) == 1  # the request path never aborted


class TestInputFaultsPassThrough:
    """Unparseable / non-canonical bodies are forwarded verbatim (fail open on the way in)."""

    async def test_malformed_json_is_forwarded_unchanged(self) -> None:
        up = FakeUpstream()
        eng = _engine(up)
        body = b"{not valid json"
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert result.status_code == 200
        assert up.last is not None
        assert up.last.content == body  # passed through byte-for-byte
        assert result.meta["cache"] == "off"  # never canonicalized -> never cached

    async def test_non_dict_json_is_forwarded_unchanged(self) -> None:
        up = FakeUpstream()
        eng = _engine(up)
        body = b"[1, 2, 3]"
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.last is not None
        assert up.last.content == body
        assert result.meta["cache"] == "off"


class TestSecurityFailsClosed:
    """The one asymmetry: a detected secret forwards (open) but is never cached (closed)."""

    async def test_secret_bearing_request_is_forwarded_but_not_cached(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("my key is sk-ant-api03-" + "A" * 24)
        first = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert first.status_code == 200  # request still served (fails open for availability)
        assert up.calls == 2  # never cached (fails closed for confidentiality)
        assert second.content == REAL.content
