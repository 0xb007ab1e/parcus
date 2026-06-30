"""Tests for provider-usage capture: the parser, engine plumbing, and response headers."""

from __future__ import annotations

import json

import httpx
import respx
from fastapi.testclient import TestClient

from parcus.cache import CachePolicy, NullCache, SqliteCache
from parcus.compress import LosslessCompressor
from parcus.model import Dialect, ProviderUsage
from parcus.proxy import create_app
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.proxy.usage import parse_usage
from parcus.redact import Redactor

# --- the pure parser ---------------------------------------------------------------------------

_ANTHROPIC_BODY = json.dumps(
    {
        "type": "message",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_input_tokens": 90,
            "cache_creation_input_tokens": 10,
        },
    }
).encode()
_OPENAI_BODY = json.dumps(
    {
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 150},
        },
    }
).encode()


class TestParseUsage:
    def test_anthropic_full(self) -> None:
        u = parse_usage(Dialect.ANTHROPIC, _ANTHROPIC_BODY)
        assert u == ProviderUsage(
            input_tokens=120, output_tokens=45, cache_read_tokens=90, cache_write_tokens=10
        )

    def test_openai_full(self) -> None:
        u = parse_usage(Dialect.OPENAI, _OPENAI_BODY)
        assert u == ProviderUsage(
            input_tokens=200, output_tokens=60, cache_read_tokens=150, cache_write_tokens=0
        )

    def test_anthropic_without_cache_fields_defaults_zero(self) -> None:
        body = json.dumps({"usage": {"input_tokens": 10, "output_tokens": 5}}).encode()
        u = parse_usage(Dialect.ANTHROPIC, body)
        assert u is not None
        assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens) == (10, 0, 0)

    def test_openai_without_details(self) -> None:
        body = json.dumps({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode()
        u = parse_usage(Dialect.OPENAI, body)
        assert u is not None and u.cache_read_tokens == 0 and u.input_tokens == 7

    def test_no_usage_object_returns_none(self) -> None:
        assert parse_usage(Dialect.ANTHROPIC, b'{"type":"message"}') is None

    def test_malformed_json_returns_none(self) -> None:
        assert parse_usage(Dialect.ANTHROPIC, b"{not json") is None

    def test_non_dict_body_returns_none(self) -> None:
        assert parse_usage(Dialect.OPENAI, b"[1,2,3]") is None

    def test_unknown_dialect_returns_none(self) -> None:
        assert parse_usage(Dialect.UNKNOWN, _ANTHROPIC_BODY) is None

    def test_tolerates_wrong_types(self) -> None:
        # Strings / nulls / bools coerce to 0 rather than raising.
        body = json.dumps(
            {"usage": {"input_tokens": "x", "output_tokens": None, "cache_read_input_tokens": True}}
        ).encode()
        u = parse_usage(Dialect.ANTHROPIC, body)
        assert u == ProviderUsage(0, 0, 0, 0)

    def test_float_truncates(self) -> None:
        body = json.dumps({"usage": {"prompt_tokens": 12.9}}).encode()
        u = parse_usage(Dialect.OPENAI, body)
        assert u is not None and u.input_tokens == 12


# --- engine plumbing ---------------------------------------------------------------------------


class _Upstream:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.calls = 0

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.calls += 1
        return UpstreamResponse(200, (("content-type", "application/json"),), self._body)


def _engine(upstream: _Upstream, **kw: object) -> ProxyEngine:
    return ProxyEngine(
        upstream=upstream,
        compressor=LosslessCompressor(),
        cache=kw.get("cache", NullCache()),  # type: ignore[arg-type]
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            cache_enabled=bool(kw.get("cache_enabled", False)),
        ),
    )


def _anthropic_req() -> bytes:
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}
    return json.dumps(body).encode()


class TestEngineUsageCapture:
    async def test_forwarded_response_usage_is_captured(self) -> None:
        eng = _engine(_Upstream(_ANTHROPIC_BODY))
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_req())
        usage = result.meta["upstream_usage"]
        assert isinstance(usage, ProviderUsage)
        assert usage.input_tokens == 120 and usage.cache_read_tokens == 90

    async def test_usage_absent_when_provider_reports_none(self) -> None:
        eng = _engine(_Upstream(b'{"type":"message"}'))
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_req())
        assert "upstream_usage" not in result.meta

    async def test_cache_hit_has_no_upstream_usage(self) -> None:
        eng = _engine(_Upstream(_ANTHROPIC_BODY), cache=SqliteCache(), cache_enabled=True)
        body = _anthropic_req()
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)  # miss -> stores
        hit = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert hit.meta["cache"] == "hit"
        assert "upstream_usage" not in hit.meta  # no upstream call -> no usage

    async def test_malformed_body_fails_open(self) -> None:
        eng = _engine(_Upstream(b"{not json"))
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_req())
        assert result.status_code == 200  # response still served
        assert "upstream_usage" not in result.meta


class TestUsageEventField:
    async def test_savings_event_carries_usage(self) -> None:
        class Spy:
            def __init__(self) -> None:
                self.events: list[object] = []

            def record(self, event: object) -> None:
                self.events.append(event)

        spy = Spy()
        eng = ProxyEngine(
            upstream=_Upstream(_ANTHROPIC_BODY),
            compressor=LosslessCompressor(),
            cache=NullCache(),
            redactor=Redactor(),
            policy=CachePolicy(),
            config=EngineConfig(
                anthropic_upstream="https://a.test", openai_upstream="https://o.test"
            ),
            metrics=spy,
        )
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_req())
        event = spy.events[0]
        assert event.upstream_usage is not None  # type: ignore[attr-defined]
        assert event.upstream_usage.cache_read_tokens == 90  # type: ignore[attr-defined]
        assert event.to_dict()["upstream_usage"]["input_tokens"] == 120  # type: ignore[attr-defined]


# --- response headers --------------------------------------------------------------------------


class _AppUpstream:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        return UpstreamResponse(200, (("content-type", "application/json"),), self._body)


def _app_engine(body: bytes) -> ProxyEngine:
    return ProxyEngine(
        upstream=_AppUpstream(body),
        compressor=LosslessCompressor(),
        cache=NullCache(),
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            cache_enabled=False,
        ),
    )


class TestUsageHeaders:
    def test_upstream_usage_headers_rendered(self) -> None:
        with TestClient(create_app(_app_engine(_ANTHROPIC_BODY))) as client:
            r = client.post(
                "/v1/messages",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "k"},
            )
        assert r.headers["x-parcus-upstream-input-tokens"] == "120"
        assert r.headers["x-parcus-upstream-output-tokens"] == "45"
        assert r.headers["x-parcus-upstream-cache-read-tokens"] == "90"
        assert r.headers["x-parcus-upstream-cache-write-tokens"] == "10"

    def test_no_usage_headers_when_provider_reports_none(self) -> None:
        with TestClient(create_app(_app_engine(b'{"type":"message"}'))) as client:
            r = client.post(
                "/v1/messages",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "k"},
            )
        assert "x-parcus-upstream-input-tokens" not in r.headers


@respx.mock
def test_streaming_path_has_no_usage_headers_yet() -> None:
    # Streaming requests bypass the engine (compression/usage) today — documents current behaviour
    # ahead of the streaming-request work (#1).
    respx.post("https://a.test/v1/messages").mock(
        return_value=httpx.Response(
            200, content=b"event: ping\n\n", headers={"content-type": "text/event-stream"}
        )
    )
    with TestClient(create_app(_app_engine(b""))) as client:
        r = client.post(
            "/v1/messages",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "k"},
        )
    assert r.headers["x-parcus-cache"] == "stream-bypass"
    assert "x-parcus-upstream-input-tokens" not in r.headers
