"""A bounded, TTL'd "have I seen this prefix before?" set for repeat-aware cache injection.

M1b prompt-cache injection pays the provider's cache-**write** premium (~1.25x) on the first turn
and recoups it as cache-**reads** (~0.1x) on later turns. For a prefix sent once and never
repeated, that write is a small net loss. This set lets the engine inject only once a prefix has
actually been **seen before** (a repeat is likely), keeping injection never-cost-more in
expectation. See issue #56 and ``docs/design/token-reduction-roadmap.md`` §2.1.

Stores only a **hash** of the prefix (never prefix text — confidential-by-default, master §5),
scoped per tenant, bounded by an LRU cap, and expired around the provider's cache window. Pure
except for an injected clock; every consumer treats it as best-effort (fail-open).
"""

from __future__ import annotations

from collections import OrderedDict

from parcus.cache.clock import SystemClock
from parcus.ports import ClockPort

__all__ = ["PrefixSeenSet"]


class PrefixSeenSet:
    """Tracks recently-seen prefix hashes (per tenant) to gate repeat-aware injection."""

    def __init__(
        self,
        *,
        max_entries: int = 4096,
        ttl_seconds: float = 300.0,
        clock: ClockPort | None = None,
    ) -> None:
        """Configure the bounded, TTL'd seen-set.

        Args:
            max_entries: LRU cap on retained hashes (bounds memory); oldest are evicted past it.
            ttl_seconds: A prior sighting older than this counts as *not* seen (tie to the provider
                cache window — Anthropic ephemeral is ~5 min).
            clock: Injected time source (for testability); defaults to the system clock.
        """
        self._max = max(1, max_entries)
        self._ttl = ttl_seconds
        self._clock = clock or SystemClock()
        self._seen: OrderedDict[str, float] = OrderedDict()

    def record_and_check(self, digest: str, *, tenant: str = "") -> bool:
        """Record a sighting of ``digest`` and return whether it was already seen within the TTL.

        First sighting (or one older than the TTL) → ``False`` (record it). A repeat within the TTL
        → ``True``. Always refreshes the entry's recency and enforces the LRU cap.
        """
        now = self._clock.now()
        key = f"{tenant}\x00{digest}" if tenant else digest
        prior = self._seen.pop(key, None)
        seen_recently = prior is not None and (now - prior) <= self._ttl
        self._seen[key] = now  # (re)insert as most-recently-seen
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)  # evict least-recently-seen
        return seen_recently
