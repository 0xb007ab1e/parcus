"""Runtime observability: per-request token-savings metrics.

Distinct from :mod:`parcus.eval` (offline measurement). Events carry counts/metadata only —
never prompt or response content — and recording is best-effort (never breaks a request).
"""

from parcus.obs.events import SavingsEvent, StageStat
from parcus.obs.prometheus import render_prometheus
from parcus.obs.report import render_stats
from parcus.obs.sinks import AggregateSink, LoggingSink, MetricsSink, MultiSink, NullSink
from parcus.obs.store import SqliteMetricsSink

__all__ = [
    "AggregateSink",
    "LoggingSink",
    "MetricsSink",
    "MultiSink",
    "NullSink",
    "SavingsEvent",
    "SqliteMetricsSink",
    "StageStat",
    "render_prometheus",
    "render_stats",
]
