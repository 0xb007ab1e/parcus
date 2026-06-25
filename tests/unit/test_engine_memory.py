"""Tests for Track B memory wired into the engine (ingest + compaction, behind flags)."""

from __future__ import annotations

import json

from parcus import cli
from parcus.cache import CachePolicy, NullCache
from parcus.compress import LosslessCompressor
from parcus.config import Settings
from parcus.model import CanonicalRequest
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor

OK = UpstreamResponse(200, (("content-type", "application/json"),), b"{}")


class FakeUpstream:
    def __init__(self) -> None:
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.last = request
        return OK


class FakeMemory:
    def __init__(self, snippets: tuple[str, ...] = (), *, raise_on_ingest: bool = False) -> None:
        self._snippets = snippets
        self._raise = raise_on_ingest
        self.ingested: list[CanonicalRequest] = []

    def ingest(self, request: CanonicalRequest) -> None:
        if self._raise:
            raise RuntimeError("boom")
        self.ingested.append(request)

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        return self._snippets[:limit]


def _engine(upstream: FakeUpstream, memory: object | None = None, **cfg: object) -> ProxyEngine:
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
            **cfg,  # type: ignore[arg-type]
        ),
        memory=memory,  # type: ignore[arg-type]
    )


def _long_anthropic(n: int = 12) -> bytes:
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message number {i} here"}
        for i in range(n)
    ]
    return json.dumps({"model": "claude-x", "messages": messages}).encode()


class TestMemoryWiring:
    async def test_disabled_by_default_does_not_ingest(self) -> None:
        memory = FakeMemory(("x",))
        eng = _engine(FakeUpstream(), memory)  # memory_enabled defaults False
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _long_anthropic())
        assert memory.ingested == []
        assert result.meta["memory"] == "off"

    async def test_ingest_mode_records_without_changing_request(self) -> None:
        memory = FakeMemory(("x",))
        eng = _engine(FakeUpstream(), memory, memory_enabled=True)  # inject off
        body = _long_anthropic()
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert len(memory.ingested) == 1
        assert result.meta["memory"] == "ingest"
        sent = json.loads(eng._upstream.last.content)  # type: ignore[union-attr]
        assert len(sent["messages"]) == 12  # unchanged (no compaction)

    async def test_inject_mode_compacts_long_request(self) -> None:
        up = FakeUpstream()
        eng = _engine(
            up,
            FakeMemory(("earlier fact one", "earlier fact two")),
            memory_enabled=True,
            memory_inject=True,
            memory_min_messages=8,
            memory_keep_recent=4,
        )
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _long_anthropic())
        sent = json.loads(up.last.content)
        assert len(sent["messages"]) < 12  # older turns dropped
        assert "Relevant earlier context" in sent["messages"][0]["content"]
        assert result.meta["memory"] == "compact"
        assert result.meta["tokens_after"] < result.meta["tokens_before"]

    async def test_fails_open_when_ingest_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, FakeMemory(raise_on_ingest=True), memory_enabled=True, memory_inject=True)
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _long_anthropic())
        assert result.meta["memory"] == "off"  # failed open
        assert result.status_code == 200  # request still forwarded


class TestCliMemoryWiring:
    def test_build_engine_wires_memory_when_enabled(self) -> None:
        eng = cli.build_engine(Settings(_env_file=None, cache=False, metrics=False, memory=True))
        assert eng._memory_provider.for_tenant("") is not None

    def test_build_engine_has_no_memory_by_default(self) -> None:
        eng = cli.build_engine(Settings(_env_file=None, cache=False, metrics=False))
        assert eng._memory_provider.for_tenant("") is None

    def test_build_engine_uses_per_tenant_provider_in_multi_tenant_mode(self) -> None:
        from parcus.memory import PerTenantMemoryProvider

        eng = cli.build_engine(
            Settings(_env_file=None, cache=False, metrics=False, memory=True, multi_tenant=True)
        )
        assert isinstance(eng._memory_provider, PerTenantMemoryProvider)
        # Distinct tenants get distinct, isolated graphs.
        assert eng._memory_provider.for_tenant("a") is not eng._memory_provider.for_tenant("b")


class TestPerTenantMemoryRouting:
    """In multi-tenant mode the engine ingests into the requesting tenant's own graph."""

    async def test_ingest_routed_to_separate_tenant_memories(self) -> None:
        from parcus.memory import PerTenantMemoryProvider

        built: list[FakeMemory] = []

        def factory() -> FakeMemory:
            mem = FakeMemory()
            built.append(mem)
            return mem

        eng = ProxyEngine(
            upstream=FakeUpstream(),
            compressor=LosslessCompressor(),
            cache=NullCache(),
            redactor=Redactor(),
            policy=CachePolicy(),
            config=EngineConfig(
                anthropic_upstream="https://a.test",
                openai_upstream="https://o.test",
                cache_enabled=False,
                memory_enabled=True,
                multi_tenant=True,
            ),
            memory_provider=PerTenantMemoryProvider(factory),  # type: ignore[arg-type]
        )
        body = _long_anthropic()
        await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-b")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        assert len(built) == 2  # one graph per distinct tenant, reused for the repeat
        # Each tenant's graph only saw its own requests.
        assert len(built[0].ingested) == 2  # tenant-a's two requests
        assert len(built[1].ingested) == 1  # tenant-b's single request
