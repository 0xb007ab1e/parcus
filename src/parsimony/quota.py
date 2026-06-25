"""Per-tenant rate limiting (token bucket) — bound cost and prevent noisy-neighbour abuse.

A hosted proxy must cap how fast any one tenant can drive upstream calls, to control cost
(OWASP LLM04 / API4 unrestricted resource consumption) and to stop one tenant degrading others
(noisy-neighbour — topic-multi-tenancy). Each tenant gets an independent **token bucket**:
``capacity`` burst tokens that refill at ``refill_per_sec``; each request consumes one. When the
bucket is empty the request is denied (the engine returns ``429`` + ``Retry-After``) and never
reaches upstream — this control **fails closed** against abuse, distinct from the optimization
path, which fails open.

Off by default (a rate of 0 disables limiting). Elapsed time comes from a **monotonic** source so
an NTP/wall-clock step can neither grant nor revoke tokens (topic-numeric-correctness). The proxy
runs on a single-threaded event loop, so each check (refill→compare→consume, no ``await`` within)
is race-free.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RateDecision", "RateLimit", "RateLimiter"]


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Token-bucket parameters: ``capacity`` burst tokens, refilled at ``refill_per_sec`` per s."""

    capacity: float
    refill_per_sec: float

    @classmethod
    def per_minute(cls, rate: float, burst: float = 0.0) -> RateLimit:
        """Build a limit of ``rate`` requests/minute with an optional ``burst`` capacity.

        Args:
            rate: Sustained requests per minute.
            burst: Bucket capacity (max instantaneous burst). ``0`` defaults it to one minute's
                worth (``rate``), so a tenant may burst up to its per-minute budget then sustain
                ``rate``/60 per second.

        Returns:
            The corresponding :class:`RateLimit`.
        """
        capacity = burst if burst > 0 else rate
        return cls(capacity=capacity, refill_per_sec=rate / 60.0)


@dataclass(frozen=True, slots=True)
class RateDecision:
    """Outcome of a rate check: whether allowed, and seconds until a token frees up if not."""

    allowed: bool
    retry_after: float = 0.0


@dataclass
class _Bucket:
    """Mutable per-key bucket state."""

    tokens: float
    updated: float


class RateLimiter:
    """An independent token bucket per key (tenant id).

    Args:
        limit: The per-key token-bucket parameters.
        time_source: Monotonic seconds source (injected for deterministic tests).
    """

    def __init__(self, limit: RateLimit, time_source: Callable[[], float] = time.monotonic) -> None:
        """Hold the limit, time source, and an empty per-key bucket map."""
        self._limit = limit
        self._now = time_source
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str) -> RateDecision:
        """Consume one token for ``key``; allow when available, else deny with a retry hint.

        Args:
            key: The bucket key (the tenant id).

        Returns:
            A :class:`RateDecision`. On denial, ``retry_after`` is the seconds until one token
            accrues (``0`` only in the degenerate no-refill case).
        """
        now = self._now()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self._limit.capacity, updated=now)
            self._buckets[key] = bucket
        # Refill for elapsed time (monotonic; never negative), capped at capacity.
        elapsed = max(0.0, now - bucket.updated)
        bucket.tokens = min(
            self._limit.capacity, bucket.tokens + elapsed * self._limit.refill_per_sec
        )
        bucket.updated = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return RateDecision(allowed=True)
        deficit = 1.0 - bucket.tokens
        retry_after = (
            deficit / self._limit.refill_per_sec if self._limit.refill_per_sec > 0 else 0.0
        )
        return RateDecision(allowed=False, retry_after=retry_after)
