"""Tests for the observability module (savings event + sinks)."""

from __future__ import annotations

import json
import logging

import pytest

from parcus.obs import AggregateSink, LoggingSink, MultiSink, NullSink, SavingsEvent


def _event(
    *,
    cache: str = "miss",
    tokens_before: int = 100,
    tokens_after: int = 60,
    canonicalized: bool = True,
    status_code: int = 200,
    tenant: str = "",
) -> SavingsEvent:
    return SavingsEvent(
        request_id="req-1",
        dialect="anthropic",
        cache=cache,
        canonicalized=canonicalized,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        status_code=status_code,
        duration_ms=1.5,
        tenant=tenant,
    )


class _SpySink:
    def __init__(self) -> None:
        self.events: list[SavingsEvent] = []

    def record(self, event: SavingsEvent) -> None:
        self.events.append(event)


class TestSavingsEvent:
    def test_saved_and_ratio(self) -> None:
        e = _event(tokens_before=100, tokens_after=60)
        assert e.tokens_saved == 40
        assert e.ratio == 0.4

    def test_ratio_zero_without_input(self) -> None:
        assert _event(tokens_before=0, tokens_after=0).ratio == 0.0

    def test_to_dict_is_content_free_and_stable(self) -> None:
        d = _event().to_dict()
        assert d["event"] == "savings"
        assert d["tokens_saved"] == 40
        assert set(d) == {
            "event",
            "request_id",
            "dialect",
            "cache",
            "canonicalized",
            "tokens_before",
            "tokens_after",
            "tokens_saved",
            "ratio",
            "status_code",
            "duration_ms",
            "tenant",
            "stages",
        }


class TestSinks:
    def test_null_sink_is_noop(self) -> None:
        assert NullSink().record(_event()) is None

    def test_logging_sink_emits_json(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="parcus.savings")
        LoggingSink().record(_event())
        payload = json.loads(caplog.records[-1].getMessage())
        assert payload["event"] == "savings"
        assert payload["tokens_saved"] == 40

    def test_aggregate_sink_totals(self) -> None:
        agg = AggregateSink()
        agg.record(_event(cache="hit", tokens_before=100, tokens_after=60))
        agg.record(_event(cache="miss", tokens_before=50, tokens_after=50))
        snap = agg.snapshot()
        assert snap["requests"] == 2
        assert snap["cache_hits"] == 1
        assert snap["cache_hit_rate"] == 0.5
        assert snap["tokens_saved"] == 40
        assert snap["overall_ratio"] == round(40 / 150, 4)

    def test_aggregate_empty_snapshot(self) -> None:
        snap = AggregateSink().snapshot()
        assert snap["requests"] == 0
        assert snap["overall_ratio"] == 0.0
        assert snap["cache_hit_rate"] == 0.0
        assert snap["by_tenant"] == {}

    def test_aggregate_by_tenant_attribution(self) -> None:
        agg = AggregateSink()
        agg.record(_event(tenant="t1", tokens_before=100, tokens_after=60))
        agg.record(_event(tenant="t1", tokens_before=100, tokens_after=80))
        agg.record(_event(tenant="t2", tokens_before=50, tokens_after=50))
        agg.record(_event(tenant=""))  # single-tenant noise is excluded
        by_tenant = agg.snapshot()["by_tenant"]
        assert set(by_tenant) == {"t1", "t2"}
        assert by_tenant["t1"]["requests"] == 2
        assert by_tenant["t1"]["tokens_saved"] == 60
        assert by_tenant["t2"]["reduction"] == 0.0

    def test_multi_sink_fans_out(self) -> None:
        a, b = _SpySink(), _SpySink()
        MultiSink([a, b]).record(_event())
        assert len(a.events) == 1
        assert len(b.events) == 1
