"""SSE / streaming-fidelity tests for the proxy passthrough.

A token-thrift proxy must be invisible on the streaming path too: it relays the provider's
Server-Sent-Events **byte-for-byte and in order**, delivers them **incrementally** (it must not
buffer the whole stream before responding — that would break every streaming harness and defeat
backpressure), drops only the unsafe hop-by-hop headers, marks the response ``stream-bypass``,
and closes the upstream when done.

These drive ``_stream_passthrough`` directly with an injected fake streaming client whose chunks
are released through ``asyncio`` gates — so incremental delivery and non-buffering are asserted
deterministically, without real sockets or transport-level buffering ambiguity. (The route-level
wiring is covered in ``test_proxy_app.py``.)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest

from parcus.cache import CachePolicy, NullCache
from parcus.compress import LosslessCompressor
from parcus.proxy.app import _is_stream, _stream_passthrough
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor

_SSE = [
    b"event: message_start\n\n",
    b'data: {"type":"content_block_delta","delta":{"text":"hel"}}\n\n',
    b'data: {"type":"content_block_delta","delta":{"text":"lo"}}\n\n',
    b"event: message_stop\n\n",
]


class _FakeStreamResponse:
    """A stand-in for httpx's streaming response, with optional per-chunk release gates."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        chunks: list[bytes],
        *,
        gates: list[asyncio.Event] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers)
        self._chunks = chunks
        self._gates = gates
        self.closed = False

    async def aiter_raw(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self._chunks):
            if self._gates is not None:
                await self._gates[index].wait()  # block until the test releases this chunk
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _FakeStreamClient:
    """An injected ``app.state.stream_client``: records the request and streams a reply."""

    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response
        self.request: SimpleNamespace | None = None

    def build_request(
        self,
        method: str,
        url: str,
        headers: list[tuple[str, str]] | None = None,
        content: bytes | None = None,
    ) -> SimpleNamespace:
        self.request = SimpleNamespace(method=method, url=url, headers=headers, content=content)
        return self.request

    async def send(self, request: SimpleNamespace, stream: bool = False) -> _FakeStreamResponse:
        assert stream is True  # the passthrough must request a streaming send
        return self._response


class _NoUpstream:
    """Upstream that must never be called — prepare_stream is request-side only (no upstream)."""

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        raise AssertionError("prepare_stream must not call the upstream")


def _engine() -> ProxyEngine:
    """A real engine for the request-side prepare_stream (the fake client does the streaming)."""
    return ProxyEngine(
        upstream=_NoUpstream(),
        compressor=LosslessCompressor(),
        cache=NullCache(),
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(anthropic_upstream="https://a.test", openai_upstream="https://o.test"),
    )


def _app_with(client: _FakeStreamClient) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(stream_client=client))


async def _passthrough(client, headers=None, body=b'{"stream":true}', path="/v1/messages"):
    hdrs = [("x-api-key", "k")] if headers is None else headers
    return await _stream_passthrough(_app_with(client), _engine(), "POST", path, hdrs, body)


class TestByteFidelity:
    async def test_relays_chunks_byte_exact_and_ordered(self) -> None:
        resp = _FakeStreamResponse(200, {"content-type": "text/event-stream"}, list(_SSE))
        out = await _passthrough(_FakeStreamClient(resp))
        assert out.status_code == 200
        received = [chunk async for chunk in out.body_iterator]
        assert received == _SSE  # same chunks, same order, NOT coalesced
        assert b"".join(received) == b"".join(_SSE)  # byte-for-byte

    async def test_empty_stream_relays_nothing(self) -> None:
        resp = _FakeStreamResponse(200, {"content-type": "text/event-stream"}, [])
        out = await _passthrough(_FakeStreamClient(resp))
        assert [chunk async for chunk in out.body_iterator] == []


class TestIncrementalDelivery:
    async def test_does_not_buffer_before_responding(self) -> None:
        # All chunks gated shut. A buffering proxy would block here (reading the whole body before
        # returning); the passthrough must return the StreamingResponse promptly regardless.
        gates = [asyncio.Event() for _ in _SSE]
        resp = _FakeStreamResponse(200, {}, list(_SSE), gates=gates)
        out = await asyncio.wait_for(_passthrough(_FakeStreamClient(resp)), 2.0)
        for gate in gates:
            gate.set()
        received = [chunk async for chunk in out.body_iterator]
        assert received == _SSE

    async def test_streams_one_chunk_at_a_time(self) -> None:
        # Release chunks one by one and confirm each is delivered before the next is produced —
        # true incremental streaming with consumer-driven backpressure.
        gates = [asyncio.Event() for _ in _SSE]
        resp = _FakeStreamResponse(200, {}, list(_SSE), gates=gates)
        out = await _passthrough(_FakeStreamClient(resp))
        iterator = out.body_iterator.__aiter__()
        for index, expected in enumerate(_SSE):
            gates[index].set()
            chunk = await asyncio.wait_for(iterator.__anext__(), 2.0)
            assert chunk == expected
        with pytest.raises(StopAsyncIteration):
            await iterator.__anext__()


class TestHeadersAndLifecycle:
    async def test_relays_status_and_headers_with_bypass_marker(self) -> None:
        resp = _FakeStreamResponse(
            206,
            {
                "content-type": "text/event-stream",
                "x-request-id": "abc",
                "content-length": "5",  # hop-by-hop / framing headers must be dropped
                "transfer-encoding": "chunked",
            },
            [b"data: x\n\n"],
        )
        out = await _passthrough(_FakeStreamClient(resp))
        present = {k.lower() for k in out.headers}
        assert out.status_code == 206
        assert "content-length" not in present
        assert "transfer-encoding" not in present
        assert out.headers["x-request-id"] == "abc"
        assert out.headers["content-type"] == "text/event-stream"
        assert out.headers["x-parcus-cache"] == "stream-bypass"

    async def test_forwarded_request_drops_unsafe_headers(self) -> None:
        client = _FakeStreamClient(_FakeStreamResponse(200, {}, [b"x"]))
        headers = [
            ("host", "proxy.local"),
            ("content-length", "14"),
            ("accept-encoding", "gzip"),
            ("x-api-key", "k"),
        ]
        await _passthrough(client, headers=headers)
        assert client.request is not None
        forwarded = {k.lower() for k, _ in client.request.headers}
        assert "host" not in forwarded
        assert "content-length" not in forwarded
        assert "accept-encoding" not in forwarded
        assert "x-api-key" in forwarded  # the credential is still forwarded to the provider

    async def test_closes_upstream_when_done(self) -> None:
        resp = _FakeStreamResponse(200, {}, [b"x"])
        out = await _passthrough(_FakeStreamClient(resp))
        assert out.background is not None  # a close task is scheduled
        await out.background()  # Starlette runs this after sending; invoke it directly
        assert resp.closed is True


class TestRoutingAndDetection:
    async def test_unroutable_stream_returns_502_without_calling_upstream(self) -> None:
        client = _FakeStreamClient(_FakeStreamResponse(200, {}, [b"x"]))
        out = await _passthrough(client, path="/v1/models", headers=[])
        assert out.status_code == 502
        assert client.request is None  # no upstream request was ever built/sent

    def test_is_stream_detection(self) -> None:
        assert _is_stream(b'{"stream": true}') is True
        assert _is_stream(b'{"stream": false}') is False
        assert _is_stream(b'{"model": "m"}') is False
        assert _is_stream(b"not json") is False
        assert _is_stream(b"[1, 2, 3]") is False  # valid JSON but not an object


def _stream_body(content: str) -> bytes:
    return json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": content}], "stream": True}
    ).encode()


class TestStreamingRequestCompression:
    """The new behaviour (#1): the streaming *request* body is compressed before forwarding,
    while the response still streams through untouched (covered above)."""

    async def test_request_body_is_compressed_and_stream_flag_preserved(self) -> None:
        client = _FakeStreamClient(_FakeStreamResponse(200, {}, [b"data: x\n\n"]))
        out = await _passthrough(client, body=_stream_body("hi   \n\n\n\nthere"))
        assert client.request is not None
        forwarded = json.loads(client.request.content)
        # lossless: trailing whitespace stripped, 3+ newlines collapsed to one blank line
        assert forwarded["messages"][0]["content"] == "hi\n\nthere"
        assert forwarded["stream"] is True  # still a streaming request
        # request-compression now surfaces on the streamed response's headers
        before = int(out.headers["x-parcus-tokens-before"])
        after = int(out.headers["x-parcus-tokens-after"])
        assert before >= after
        assert out.headers["x-parcus-cache"] == "stream-bypass"


class TestStreamingFrontHalf:
    """Streaming requests now also get authorization + rate limiting (previously bypassed)."""

    def _engine(self, **kw: object) -> ProxyEngine:
        return ProxyEngine(
            upstream=_NoUpstream(),
            compressor=LosslessCompressor(),
            cache=NullCache(),
            redactor=Redactor(),
            policy=CachePolicy(),
            config=EngineConfig(
                anthropic_upstream="https://a.test",
                openai_upstream="https://o.test",
                multi_tenant=bool(kw.get("multi_tenant", False)),
                allowed_tenants=kw.get("allowed_tenants", frozenset()),  # type: ignore[arg-type]
            ),
            rate_limiter=kw.get("rate_limiter"),  # type: ignore[arg-type]
        )

    async def test_unlisted_tenant_gets_401_without_forwarding(self) -> None:
        eng = self._engine(multi_tenant=True, allowed_tenants=frozenset({"not-this-one"}))
        client = _FakeStreamClient(_FakeStreamResponse(200, {}, [b"x"]))
        out = await _stream_passthrough(
            _app_with(client), eng, "POST", "/v1/messages", [("x-api-key", "k")], _stream_body("hi")
        )
        assert out.status_code == 401
        assert client.request is None  # never forwarded

    async def test_over_rate_limit_gets_429_without_forwarding(self) -> None:
        from parcus.quota import RateLimit, RateLimiter

        limiter = RateLimiter(RateLimit(capacity=1, refill_per_sec=0.0), time_source=lambda: 0.0)
        eng = self._engine(rate_limiter=limiter)
        body = _stream_body("hi")
        first = _FakeStreamClient(_FakeStreamResponse(200, {}, [b"x"]))
        second = _FakeStreamClient(_FakeStreamResponse(200, {}, [b"x"]))
        h = [("x-api-key", "k")]
        r1 = await _stream_passthrough(_app_with(first), eng, "POST", "/v1/messages", h, body)
        r2 = await _stream_passthrough(_app_with(second), eng, "POST", "/v1/messages", h, body)
        assert r1.status_code == 200
        assert r2.status_code == 429
        assert second.request is None  # the shed request never reached an upstream
