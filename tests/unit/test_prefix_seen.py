"""Unit tests for the repeat-aware prefix-seen set (issue #56)."""

from __future__ import annotations

from parcus.cache.seen import PrefixSeenSet


class FakeClock:
    """A settable clock for deterministic TTL tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t


def test_first_sighting_is_not_seen() -> None:
    seen = PrefixSeenSet(clock=FakeClock())
    assert seen.record_and_check("h") is False


def test_second_sighting_within_ttl_is_seen() -> None:
    seen = PrefixSeenSet(clock=FakeClock())
    assert seen.record_and_check("h") is False
    assert seen.record_and_check("h") is True


def test_expired_sighting_counts_as_unseen() -> None:
    clock = FakeClock(0.0)
    seen = PrefixSeenSet(ttl_seconds=300.0, clock=clock)
    assert seen.record_and_check("h") is False
    clock.t = 301.0  # prior sighting now older than the TTL
    assert seen.record_and_check("h") is False
    clock.t = 302.0  # freshly recorded above → now seen
    assert seen.record_and_check("h") is True


def test_tenant_isolation() -> None:
    seen = PrefixSeenSet(clock=FakeClock())
    assert seen.record_and_check("h", tenant="a") is False
    assert seen.record_and_check("h", tenant="b") is False  # independent per tenant
    assert seen.record_and_check("h", tenant="a") is True


def test_lru_eviction_bounds_memory() -> None:
    seen = PrefixSeenSet(max_entries=2, clock=FakeClock())
    seen.record_and_check("a")  # {a}
    seen.record_and_check("b")  # {a, b}
    seen.record_and_check("c")  # over cap → evict a → {b, c}
    assert seen.record_and_check("a") is False  # a was evicted (and now evicts b)
    assert seen.record_and_check("b") is False  # b was evicted too


def test_default_clock_smoke() -> None:
    # No injected clock → real SystemClock; two back-to-back sightings are within the TTL.
    seen = PrefixSeenSet()
    assert seen.record_and_check("h") is False
    assert seen.record_and_check("h") is True
