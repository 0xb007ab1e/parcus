"""Tests for the httpx upstream adapter (mocked with respx — no real network)."""

from __future__ import annotations

import httpx
import respx

from parcus.proxy.upstream import HttpxUpstream, UpstreamRequest


@respx.mock
async def test_send_roundtrips_status_headers_and_body() -> None:
    route = respx.post("https://up.test/v1/messages").mock(
        return_value=httpx.Response(
            201, json={"ok": True}, headers={"content-type": "application/json"}
        )
    )
    upstream = HttpxUpstream()
    response = await upstream.send(
        UpstreamRequest(
            method="POST",
            url="https://up.test/v1/messages",
            headers=(("x-api-key", "k"),),
            content=b'{"a":1}',
        )
    )
    assert response.status_code == 201
    assert b"ok" in response.content
    assert any(k.lower() == "content-type" for k, _ in response.headers)
    assert route.called
    await upstream.aclose()
