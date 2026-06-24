"""Metrics sinks: where per-request :class:`SavingsEvent`s go.

A sink is anything implementing :class:`MetricsSink`. Sinks must never raise into the request
path — recording metrics is best-effort.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from parsimony.obs.events import SavingsEvent

__all__ = ["AggregateSink", "LoggingSink", "MetricsSink", "MultiSink", "NullSink"]


@runtime_checkable
class MetricsSink(Protocol):
    """Receives savings events. Implementations must not raise."""

    def record(self, event: SavingsEvent) -> None:
        """Record one savings event."""
        ...


class NullSink:
    """Discards events (used when metrics are disabled)."""

    def record(self, event: SavingsEvent) -> None:
        """Discard the event (no-op)."""
        return


class LoggingSink:
    """Emits each event as a structured JSON log line (counts/metadata only).

    Args:
        logger: Logger to use (defaults to ``parsimony.savings``).
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialise with an optional custom logger."""
        self._logger = logger or logging.getLogger("parsimony.savings")

    def record(self, event: SavingsEvent) -> None:
        """Log the event as a single JSON object (best-effort; never raises)."""
        try:
            self._logger.info("%s", json.dumps(event.to_dict()))
        except Exception:  # pragma: no cover - logging must never break the request path
            return


class AggregateSink:
    """Thread-safe in-process running totals (for a future stats endpoint / shutdown summary)."""

    def __init__(self) -> None:
        """Start all counters at zero."""
        self._lock = threading.Lock()
        self._requests = 0
        self._cache_hits = 0
        self._tokens_before = 0
        self._tokens_after = 0

    def record(self, event: SavingsEvent) -> None:
        """Fold the event into the running totals."""
        with self._lock:
            self._requests += 1
            if event.cache == "hit":
                self._cache_hits += 1
            self._tokens_before += event.tokens_before
            self._tokens_after += event.tokens_after

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy of the aggregate metrics."""
        with self._lock:
            saved = self._tokens_before - self._tokens_after
            ratio = saved / self._tokens_before if self._tokens_before > 0 else 0.0
            hit_rate = self._cache_hits / self._requests if self._requests > 0 else 0.0
            return {
                "requests": self._requests,
                "cache_hits": self._cache_hits,
                "cache_hit_rate": round(hit_rate, 4),
                "tokens_before": self._tokens_before,
                "tokens_after": self._tokens_after,
                "tokens_saved": saved,
                "overall_ratio": round(ratio, 4),
            }


class MultiSink:
    """Fan an event out to several sinks.

    Args:
        sinks: The sinks to forward each event to, in order.
    """

    def __init__(self, sinks: Iterable[MetricsSink]) -> None:
        """Store the downstream sinks."""
        self._sinks = tuple(sinks)

    def record(self, event: SavingsEvent) -> None:
        """Forward the event to every downstream sink."""
        for sink in self._sinks:
            sink.record(event)
