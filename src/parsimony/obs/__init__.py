"""Runtime observability: per-request token-savings metrics.

Distinct from :mod:`parsimony.eval` (offline measurement). Events carry counts/metadata only —
never prompt or response content — and recording is best-effort (never breaks a request).
"""

from parsimony.obs.events import SavingsEvent, StageStat
from parsimony.obs.prometheus import render_prometheus
from parsimony.obs.report import render_stats
from parsimony.obs.sinks import AggregateSink, LoggingSink, MetricsSink, MultiSink, NullSink
from parsimony.obs.store import SqliteMetricsSink

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
