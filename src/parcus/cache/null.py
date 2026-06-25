"""A no-op cache for when caching is disabled."""

from __future__ import annotations

from parcus.model import CachedResponse

__all__ = ["NullCache"]


class NullCache:
    """Never stores and always misses. Implements :class:`parcus.ports.CachePort`."""

    def get(self, key: str, *, tenant: str = "") -> CachedResponse | None:
        """Always return ``None`` (cache miss)."""
        return None

    def put(self, key: str, value: CachedResponse, ttl_seconds: int, *, tenant: str = "") -> None:
        """Discard the entry (no-op)."""
        return
