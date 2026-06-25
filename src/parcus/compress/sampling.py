"""Deterministic sampling for the compressors' invariant self-check.

The self-check (re-verifying the invariant per request) is O(tokens); at high throughput you
may want to run it on only a fraction of requests. :class:`VerifySampler` picks ~``rate`` of
calls **deterministically** (no RNG, so behaviour is reproducible and testable): ``rate=1.0``
always verifies, ``rate=0.0`` never does, ``rate=0.5`` verifies every other call.
"""

from __future__ import annotations

__all__ = ["VerifySampler"]


class VerifySampler:
    """Decides per call whether to run the invariant self-check.

    Args:
        rate: Fraction of calls to verify, clamped to ``[0.0, 1.0]``.
    """

    def __init__(self, rate: float = 1.0) -> None:
        """Initialise with the sampling rate."""
        self._rate = min(1.0, max(0.0, rate))
        self._calls = 0
        self._verified = 0

    def should_verify(self) -> bool:
        """Return whether this call should verify (advances the internal counters)."""
        self._calls += 1
        if self._verified < self._calls * self._rate:
            self._verified += 1
            return True
        return False
