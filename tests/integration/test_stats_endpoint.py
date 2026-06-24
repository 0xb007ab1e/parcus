"""Tests for the proxy's local JSON stats endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from parsimony.cache import CachePolicy, NullCache
from parsimony.compress import LosslessCompressor
from parsimony.proxy import create_app
from parsimony.proxy.engine import EngineConfig, ProxyEngine
from parsimony.proxy.upstream import UpstreamRequest, UpstreamResponse
from parsimony.redact import Redactor


class _FakeUpstream:
    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        return UpstreamResponse(200, (), b"{}")


class _FakeStats:
    def snapshot(self) -> dict[str, Any]:
        return {"requests": 7, "stages": {"lossless": {"reduction": 0.2}}}


def _engine() -> ProxyEngine:
    return ProxyEngine(
        upstream=_FakeUpstream(),
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


def test_stats_endpoint_returns_snapshot() -> None:
    with TestClient(create_app(_engine(), stats_source=_FakeStats())) as client:
        response = client.get("/__parsimony__/stats")
    assert response.status_code == 200
    assert response.json()["requests"] == 7  # served locally, not forwarded upstream


def test_stats_endpoint_empty_without_source() -> None:
    with TestClient(create_app(_engine())) as client:
        response = client.get("/__parsimony__/stats")
    assert response.status_code == 200
    assert response.json() == {}
