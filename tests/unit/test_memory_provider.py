"""Unit tests for per-tenant memory provisioning and isolation."""

from __future__ import annotations

from parsimony.memory import (
    GraphMemory,
    PerTenantMemoryProvider,
    SharedMemoryProvider,
)
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _request(text: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model=None,
        messages=(Message(role=Role.USER, spans=(Span(text),)),),
    )


class TestSharedMemoryProvider:
    def test_returns_same_instance_for_every_tenant(self) -> None:
        mem = GraphMemory()
        provider = SharedMemoryProvider(mem)
        assert provider.for_tenant("a") is mem
        assert provider.for_tenant("b") is mem

    def test_none_when_memory_disabled(self) -> None:
        assert SharedMemoryProvider(None).for_tenant("a") is None


class TestPerTenantMemoryProvider:
    def test_distinct_instance_per_tenant(self) -> None:
        provider = PerTenantMemoryProvider(lambda: GraphMemory())
        assert provider.for_tenant("a") is not provider.for_tenant("b")

    def test_same_instance_for_same_tenant(self) -> None:
        provider = PerTenantMemoryProvider(lambda: GraphMemory())
        assert provider.for_tenant("a") is provider.for_tenant("a")

    def test_builds_lazily_one_per_tenant(self) -> None:
        built = 0

        def factory() -> GraphMemory:
            nonlocal built
            built += 1
            return GraphMemory()

        provider = PerTenantMemoryProvider(factory)
        provider.for_tenant("a")
        provider.for_tenant("a")  # cached — no rebuild
        provider.for_tenant("b")
        assert built == 2

    def test_one_tenant_cannot_retrieve_anothers_context(self) -> None:
        # The isolation guarantee: ingest into tenant A; tenant B's graph never surfaces it.
        provider = PerTenantMemoryProvider(lambda: GraphMemory())
        provider.for_tenant("a").ingest(_request("the deploy key rotates every tuesday"))
        b_snippets = provider.for_tenant("b").relevant("deploy key rotation", limit=5)
        assert all("deploy key" not in s for s in b_snippets)
        # Sanity: A *can* retrieve its own content.
        a_snippets = provider.for_tenant("a").relevant("deploy key rotation", limit=5)
        assert any("deploy key" in s for s in a_snippets)
