"""Runtime observability: per-request token-savings metrics.

Distinct from :mod:`parsimony.eval` (offline measurement). Events carry counts/metadata only —
never prompt or response content — and recording is best-effort (never breaks a request).
"""

from parsimony.obs.events import SavingsEvent
from parsimony.obs.sinks import AggregateSink, LoggingSink, MetricsSink, MultiSink, NullSink

__all__ = [
    "AggregateSink",
    "LoggingSink",
    "MetricsSink",
    "MultiSink",
    "NullSink",
    "SavingsEvent",
]
