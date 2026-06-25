"""Unit tests for the per-tenant token-bucket rate limiter."""

from __future__ import annotations

from parsimony.quota import RateLimit, RateLimiter


class _FakeClock:
    """A controllable monotonic source."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_per_minute_factory_defaults_burst_to_one_minute() -> None:
    limit = RateLimit.per_minute(60)
    assert limit.capacity == 60
    assert limit.refill_per_sec == 1.0  # 60/min = 1/s


def test_per_minute_factory_respects_explicit_burst() -> None:
    limit = RateLimit.per_minute(60, burst=10)
    assert limit.capacity == 10


def test_allows_up_to_capacity_then_denies() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(RateLimit(capacity=3, refill_per_sec=1.0), time_source=clock)
    assert [limiter.check("t").allowed for _ in range(3)] == [True, True, True]
    denied = limiter.check("t")
    assert denied.allowed is False
    assert denied.retry_after == 1.0  # one token accrues in 1s at 1/s


def test_refills_over_time() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(RateLimit(capacity=1, refill_per_sec=1.0), time_source=clock)
    assert limiter.check("t").allowed is True
    assert limiter.check("t").allowed is False  # bucket empty
    clock.advance(1.0)
    assert limiter.check("t").allowed is True  # refilled one token


def test_refill_capped_at_capacity() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(RateLimit(capacity=2, refill_per_sec=1.0), time_source=clock)
    clock.advance(100.0)  # would overflow without the cap
    assert limiter.check("t").allowed is True
    assert limiter.check("t").allowed is True
    assert limiter.check("t").allowed is False  # only capacity (2) available, not 100


def test_tenants_are_independent() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(RateLimit(capacity=1, refill_per_sec=1.0), time_source=clock)
    assert limiter.check("a").allowed is True
    assert limiter.check("a").allowed is False  # tenant a exhausted
    assert limiter.check("b").allowed is True  # tenant b unaffected


def test_zero_refill_retry_after_is_safe() -> None:
    # Degenerate config (no refill): denial reports retry_after 0 rather than dividing by zero.
    limiter = RateLimiter(RateLimit(capacity=1, refill_per_sec=0.0), time_source=_FakeClock())
    assert limiter.check("t").allowed is True
    denied = limiter.check("t")
    assert denied.allowed is False
    assert denied.retry_after == 0.0
