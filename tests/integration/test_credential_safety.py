"""Security: the inbound provider credential must never leak into observable/persisted surfaces.

The provider API key is the crown jewel (master §1, workflow-secrets). It legitimately travels
*forward* to the provider, but it must never appear in anything parcus exposes or stores: not in
metrics/telemetry, not in response headers, and not in the cache file. The tenant id derived from
it is a one-way, content-free digest (BOLA defence — `parcus.tenant`), never the credential.

These are runnable regression guards for that guarantee (the companion DAST scaffold in
``qa/dast/`` is the external, on-demand check).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from parcus.cache import CachePolicy, SqliteCache
from parcus.compress import LosslessCompressor
from parcus.obs import SavingsEvent
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor
from parcus.tenant import derive_tenant

CRED = "SECRET-CRED-9Z9Z9Z-do-not-leak"
_RESP = UpstreamResponse(200, (("content-type", "application/json"),), b'{"answer":"ok"}')


class _FakeUpstream:
    def __init__(self) -> None:
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.last = request
        return _RESP


class _SpySink:
    def __init__(self) -> None:
        self.events: list[SavingsEvent] = []

    def record(self, event: SavingsEvent) -> None:
        self.events.append(event)


def _engine(upstream: _FakeUpstream, *, cache: SqliteCache, spy: _SpySink) -> ProxyEngine:
    return ProxyEngine(
        upstream=upstream,
        compressor=LosslessCompressor(),
        cache=cache,
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            multi_tenant=True,  # tenant derived from the credential — the most leak-prone mode
        ),
        metrics=spy,
    )


def _body() -> bytes:
    return json.dumps(
        {"model": "claude-x", "messages": [{"role": "user", "content": "summarize this"}]}
    ).encode()


async def _run(cache: SqliteCache, spy: _SpySink) -> tuple[_FakeUpstream, object]:
    up = _FakeUpstream()
    eng = _engine(up, cache=cache, spy=spy)
    result = await eng.handle("POST", "/v1/messages", [("x-api-key", CRED)], _body())
    return up, result


class TestCredentialNeverLeaks:
    async def test_not_in_metrics_event(self) -> None:
        spy = _SpySink()
        await _run(SqliteCache(), spy)
        assert spy.events, "expected a savings event"
        # The whole event, every field stringified, must not contain the credential.
        blob = json.dumps(dataclasses.asdict(spy.events[0]), default=str)
        assert CRED not in blob

    async def test_not_in_response_headers(self) -> None:
        _, result = await _run(SqliteCache(), _SpySink())
        for key, value in result.headers:
            assert CRED not in key and CRED not in value

    async def test_not_in_cache_file(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        cache = SqliteCache(str(db))
        await _run(cache, _SpySink())
        raw = db.read_bytes()
        assert _RESP.content in raw, "sanity: the response should have been cached"
        assert CRED.encode() not in raw  # ...but never the credential

    async def test_tenant_is_a_nonidentifying_digest(self) -> None:
        spy = _SpySink()
        await _run(SqliteCache(), spy)
        tenant = spy.events[0].tenant
        assert tenant and tenant != CRED  # derived, present, and not the raw credential
        assert tenant == derive_tenant([("x-api-key", CRED)])  # deterministic, one-way digest
        assert CRED not in tenant


class TestCredentialStillReachesProvider:
    async def test_credential_is_forwarded_upstream(self) -> None:
        # The flip side: the credential MUST still travel forward (parcus is a transparent proxy).
        up, _ = await _run(SqliteCache(), _SpySink())
        assert up.last is not None
        assert ("x-api-key", CRED) in up.last.headers
