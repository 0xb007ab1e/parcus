"""Tests for the proxy's local JSON stats endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from parcus.cache import CachePolicy, NullCache
from parcus.compress import LosslessCompressor
from parcus.proxy import create_app
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor


class _FakeUpstream:
    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        return UpstreamResponse(200, (), b"{}")


class _FakeStats:
    def snapshot(self) -> dict[str, Any]:
        return {
            "requests": 7,
            "cache_hits": 2,
            "cache_hit_rate": 0.2857,
            "tokens_before": 100,
            "tokens_after": 70,
            "tokens_saved": 30,
            "overall_ratio": 0.3,
            "stages": {
                "lossless": {
                    "tokens_before": 100,
                    "tokens_after": 80,
                    "tokens_saved": 20,
                    "reduction": 0.2,
                    "accuracy": 1.0,
                    "checked": 7,
                }
            },
            "evals": {},
        }


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
        response = client.get("/__parcus__/stats")
    assert response.status_code == 200
    assert response.json()["requests"] == 7  # served locally, not forwarded upstream


def test_stats_endpoint_empty_without_source() -> None:
    with TestClient(create_app(_engine())) as client:
        response = client.get("/__parcus__/stats")
    assert response.status_code == 200
    assert response.json() == {}


def test_health_endpoint() -> None:
    with TestClient(create_app(_engine())) as client:
        response = client.get("/__parcus__/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_metrics_endpoint_prometheus_format() -> None:
    with TestClient(create_app(_engine(), stats_source=_FakeStats())) as client:
        response = client.get("/__parcus__/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "parcus_requests_total 7" in body
    assert 'parcus_stage_reduction_ratio{stage="lossless"} 0.2' in body
