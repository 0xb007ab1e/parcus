"""Tests for the proxy engine: compression, caching, routing, and fail-open behaviour."""

from __future__ import annotations

import json

from parsimony.cache import CachePolicy, SqliteCache
from parsimony.compress import LosslessCompressor
from parsimony.model import CanonicalRequest, CompressionStats
from parsimony.proxy.engine import EngineConfig, ProxyEngine
from parsimony.proxy.upstream import UpstreamRequest, UpstreamResponse
from parsimony.redact import Redactor

OK = UpstreamResponse(200, (("content-type", "application/json"),), b'{"r":1}')


class FakeUpstream:
    """Records the forwarded request and returns a canned response."""

    def __init__(self, response: UpstreamResponse = OK) -> None:
        self._response = response
        self.calls = 0
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.calls += 1
        self.last = request
        return self._response


class BoomCompressor:
    """A compressor that always raises (to exercise the engine's fail-open path)."""

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        raise RuntimeError("boom")


def _engine(upstream: FakeUpstream, **kw: object) -> ProxyEngine:
    return ProxyEngine(
        upstream=upstream,
        compressor=kw.get("compressor", LosslessCompressor()),  # type: ignore[arg-type]
        cache=kw.get("cache", SqliteCache()),  # type: ignore[arg-type]
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            cache_enabled=bool(kw.get("cache_enabled", True)),
            multi_tenant=bool(kw.get("multi_tenant", False)),
        ),
        metrics=kw.get("metrics"),  # type: ignore[arg-type]
    )


class SpySink:
    """Records savings events for assertions."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def record(self, event: object) -> None:
        self.events.append(event)


def _anthropic(content: str, *, system: str | None = None, stream: bool = False) -> bytes:
    body: dict[str, object] = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": content}],
    }
    if system is not None:
        body["system"] = system
    if stream:
        body["stream"] = True
    return json.dumps(body).encode()


class TestForwardingAndCompression:
    async def test_compresses_mutable_text_and_preserves_headers(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        body = _anthropic("hello   \n\n\n\n\nworld   ", system="sys   \n\n\n\nmore")
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "hello\n\nworld"
        assert sent["system"] == "sys\n\nmore"
        assert result.meta["tokens_before"] >= result.meta["tokens_after"]
        assert ("x-api-key", "k") in up.last.headers

    async def test_passthrough_when_not_canonicalizable(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        body = json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            }
        ).encode()
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.last.content == body  # forwarded unchanged
        assert result.meta["cache"] == "off"

    async def test_fails_open_when_compressor_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, compressor=BoomCompressor(), cache_enabled=False)
        body = _anthropic("data   \n\n\n\n")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.last.content == body  # original forwarded despite the failure


class TestCaching:
    async def test_miss_then_hit_skips_second_upstream_call(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("please compute")
        first = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert first.meta["cache"] == "miss"
        assert second.meta["cache"] == "hit"
        assert up.calls == 1
        assert second.content == b'{"r":1}'

    async def test_credential_request_is_not_cached(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("token sk-ant-api03-" + "A" * 24)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.calls == 2  # never served from cache

    async def test_streaming_request_is_not_cached(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("hi", stream=True)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.calls == 2


class TestRouting:
    async def test_unknown_path_without_auth_returns_502(self) -> None:
        up = FakeUpstream()
        eng = _engine(up)
        result = await eng.handle("GET", "/v1/models", [], b"")
        assert result.status_code == 502
        assert up.calls == 0

    async def test_unknown_path_routed_by_authorization_header(self) -> None:
        up = FakeUpstream()
        eng = _engine(up)
        result = await eng.handle("GET", "/v1/models", [("authorization", "Bearer x")], b"")
        assert result.status_code == 200  # routed to OpenAI base and forwarded

    async def test_openai_chat_completions_compresses(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        body = json.dumps(
            {"model": "gpt", "messages": [{"role": "user", "content": "hi   \n\n\n\n"}]}
        ).encode()
        await eng.handle("POST", "/v1/chat/completions", [("authorization", "Bearer x")], body)
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "hi\n\n"


class TestMetrics:
    async def test_emits_one_savings_event_per_request(self) -> None:
        spy = SpySink()
        eng = _engine(FakeUpstream(), cache_enabled=False, metrics=spy)
        await eng.handle(
            "POST",
            "/v1/messages",
            [("x-request-id", "abc"), ("x-api-key", "k")],
            _anthropic("hello   \n\n\n\n"),
        )
        assert len(spy.events) == 1
        event = spy.events[0]
        assert event.request_id == "abc"  # correlation id taken from the header
        assert event.dialect == "anthropic"
        assert event.canonicalized is True
        assert event.tokens_before >= event.tokens_after
        assert event.status_code == 200
        assert event.duration_ms >= 0.0

    async def test_generates_request_id_when_header_absent(self) -> None:
        spy = SpySink()
        eng = _engine(FakeUpstream(), metrics=spy)
        await eng.handle("GET", "/v1/models", [], b"")  # unroutable -> 502, still recorded
        assert len(spy.events) == 1
        assert spy.events[0].status_code == 502
        assert spy.events[0].request_id  # a non-empty generated id


class TestMultiTenantIsolation:
    """Hosted mode: the cache is namespaced per credential-derived tenant (BOLA defence)."""

    async def test_different_tenants_never_share_cache(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=True)
        body = _anthropic("identical body across tenants")
        # Tenant A primes the cache, then tenant B sends the SAME body with a different key.
        a = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        b = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-b")], body)
        assert a.meta["cache"] == "miss"
        assert b.meta["cache"] == "miss"  # NOT served from tenant A's entry
        assert up.calls == 2  # both reached upstream — no cross-tenant leak

    async def test_same_tenant_still_hits_cache(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=True)
        body = _anthropic("same tenant repeats a request")
        first = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        assert first.meta["cache"] == "miss"
        assert second.meta["cache"] == "hit"
        assert up.calls == 1

    async def test_single_tenant_mode_unaffected(self) -> None:
        # With multi_tenant off (default), the credential does not scope the cache: a repeated
        # body hits regardless of key — correct for the local single-principal deployment.
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=False)
        body = _anthropic("local single user")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k1")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "k2")], body)
        assert second.meta["cache"] == "hit"
        assert up.calls == 1
