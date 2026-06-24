"""Clock adapters implementing :class:`parsimony.ports.ClockPort`."""

from __future__ import annotations

import time

__all__ = ["SystemClock"]


class SystemClock:
    """Wall-clock time source backed by :func:`time.time`."""

    def now(self) -> float:
        """Return the current Unix timestamp in seconds."""
        return time.time()
