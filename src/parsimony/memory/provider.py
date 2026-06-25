"""Per-tenant memory provisioning — one graph per tenant so context never crosses tenants.

The graph memory holds prior prompt content (durable facts/decisions) to inject or compact in
later turns. In hosted/multi-tenant mode a *shared* graph would be a cross-tenant leak: tenant
A's ingested context could be retrieved into tenant B's request — the memory analogue of the
cache-reuse risk (threat E1). A :class:`MemoryProvider` resolves the right memory for a tenant
*before* any ingest/retrieve, so isolation is structural rather than something each call site
must remember.

Single-tenant (local) mode uses :class:`SharedMemoryProvider` — one graph for everyone, exactly
today's behaviour. Hosted mode uses :class:`PerTenantMemoryProvider` — a separate graph per
tenant id, built lazily and cached.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from parsimony.ports import MemoryPort

__all__ = ["MemoryProvider", "PerTenantMemoryProvider", "SharedMemoryProvider"]


@runtime_checkable
class MemoryProvider(Protocol):
    """Resolves the :class:`~parsimony.ports.MemoryPort` to use for a given tenant."""

    def for_tenant(self, tenant: str) -> MemoryPort | None:
        """Return the memory for ``tenant`` (``None`` when memory is disabled)."""
        ...


class SharedMemoryProvider:
    """Return one shared memory for every tenant. Correct for single-tenant (local) mode.

    Args:
        memory: The single memory instance (or ``None`` when memory is disabled).
    """

    def __init__(self, memory: MemoryPort | None) -> None:
        """Hold the shared memory instance."""
        self._memory = memory

    def for_tenant(self, tenant: str) -> MemoryPort | None:
        """Return the shared memory regardless of tenant."""
        return self._memory


class PerTenantMemoryProvider:
    """Return a separate memory per tenant id, built lazily from a factory and cached.

    Each tenant's graph is fully isolated: ingest and retrieval for one tenant can never surface
    another tenant's content (the cross-tenant context-leak risk). The proxy runs on a
    single-threaded event loop, so the lazy build (get-then-set with no ``await`` between) is
    race-free.

    Args:
        factory: Builds a fresh :class:`~parsimony.ports.MemoryPort` for a new tenant.
    """

    def __init__(self, factory: Callable[[], MemoryPort]) -> None:
        """Hold the per-tenant memory factory and an empty cache."""
        self._factory = factory
        self._by_tenant: dict[str, MemoryPort] = {}

    def for_tenant(self, tenant: str) -> MemoryPort:
        """Return ``tenant``'s memory, creating and caching it on first use."""
        memory = self._by_tenant.get(tenant)
        if memory is None:
            memory = self._factory()
            self._by_tenant[tenant] = memory
        return memory
