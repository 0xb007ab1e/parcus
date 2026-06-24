"""Integration tests for the FastAPI proxy app (non-streaming engine path + stream bypass)."""

from __future__ import annotations

import json

import httpx
import respx
from fastapi.testclient import TestClient

from parsimony.cache import CachePolicy, NullCache
from parsimony.compress import LosslessCompressor
from parsimony.proxy import create_app
from parsimony.proxy.engine import EngineConfig, ProxyEngine
from parsimony.proxy.upstream import UpstreamRequest, UpstreamResponse
from parsimony.redact import Redactor


class FakeUpstream:
    """Records the forwarded request and returns a canned response (non-streaming path)."""

    def __init__(self, response: UpstreamResponse) -> None:
        self._response = response
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.last = request
        return self._response


def _engine(upstream: FakeUpstream) -> ProxyEngine:
    return ProxyEngine(
        upstream=upstream,
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


def test_non_streaming_request_is_proxied_and_compressed() -> None:
    up = FakeUpstream(
        UpstreamResponse(200, (("content-type", "application/json"),), b'{"ok":true}')
    )
    with TestClient(create_app(_engine(up))) as client:
        response = client.post(
            "/v1/messages",
            json={"model": "m", "messages": [{"role": "user", "content": "hi   \n\n\n\n"}]},
            headers={"x-api-key": "k"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["x-parsimony-cache"] == "off"
    assert response.headers["x-parsimony-dialect"] == "anthropic"
    assert json.loads(up.last.content)["messages"][0]["content"] == "hi\n\n"


@respx.mock
def test_streaming_request_is_passed_through() -> None:
    respx.post("https://a.test/v1/messages").mock(
        return_value=httpx.Response(
            200, content=b"event: ping\n\n", headers={"content-type": "text/event-stream"}
        )
    )
    up = FakeUpstream(UpstreamResponse(200, (), b""))  # unused on the streaming path
    with TestClient(create_app(_engine(up))) as client:
        response = client.post(
            "/v1/messages",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "k"},
        )
    assert response.status_code == 200
    assert response.headers["x-parsimony-cache"] == "stream-bypass"
    assert b"ping" in response.content


def test_unroutable_request_returns_502() -> None:
    up = FakeUpstream(UpstreamResponse(200, (), b""))
    with TestClient(create_app(_engine(up))) as client:
        response = client.get("/v1/models")  # unknown path, no auth header
    assert response.status_code == 502
