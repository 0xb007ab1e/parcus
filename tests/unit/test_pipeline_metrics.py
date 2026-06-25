"""Tests for per-stage pipeline metrics: self-checks, StageStat, aggregation, engine emission."""

from __future__ import annotations

from parcus.cache import CachePolicy, NullCache
from parcus.compress import FillerCompressor, LosslessCompressor
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
from parcus.obs import AggregateSink, SavingsEvent, StageStat
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor


def _req(text: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=(Span(text),)),),
    )


class TestStageStat:
    def test_saved_ratio_and_to_dict(self) -> None:
        s = StageStat("lossless", tokens_before=10, tokens_after=7, ok=True)
        assert s.tokens_saved == 3
        assert s.ratio == 0.3
        d = s.to_dict()
        assert d["stage"] == "lossless"
        assert d["ok"] is True
        assert d["tokens_saved"] == 3

    def test_ratio_zero_without_input(self) -> None:
        assert StageStat("x", 0, 0).ratio == 0.0


class TestCompressorSelfCheck:
    def test_lossless_reports_ok(self) -> None:
        _, stats = LosslessCompressor().compress(_req("hi   \n\n\n\nthere"))
        assert stats[0].ok is True  # whitespace-only -> invariant held

    def test_filler_reports_ok(self) -> None:
        _, stats = FillerCompressor().compress(_req("please fix this"))
        assert stats[0].ok is True  # only allow-listed fillers removed -> invariant held


class TestAggregatePerStage:
    def test_per_stage_reduction_and_accuracy(self) -> None:
        agg = AggregateSink()
        agg.record(
            SavingsEvent(
                request_id="a",
                dialect="anthropic",
                cache="miss",
                canonicalized=True,
                tokens_before=10,
                tokens_after=8,
                status_code=200,
                duration_ms=1.0,
                stages=(StageStat("lossless", 10, 8, True), StageStat("memory", 12, 10, True)),
            )
        )
        agg.record(
            SavingsEvent(
                request_id="b",
                dialect="anthropic",
                cache="miss",
                canonicalized=True,
                tokens_before=10,
                tokens_after=9,
                status_code=200,
                duration_ms=1.0,
                stages=(StageStat("lossless", 10, 9, False),),
            )
        )
        stages = agg.snapshot()["stages"]
        assert stages["lossless"]["tokens_before"] == 20
        assert stages["lossless"]["tokens_after"] == 17
        assert stages["lossless"]["checked"] == 2
        assert stages["lossless"]["accuracy"] == 0.5  # one ok, one violation
        assert stages["memory"]["accuracy"] == 1.0


class _SpySink:
    def __init__(self) -> None:
        self.events: list[SavingsEvent] = []

    def record(self, event: SavingsEvent) -> None:
        self.events.append(event)


class _FakeUpstream:
    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        return UpstreamResponse(200, (("content-type", "application/json"),), b"{}")


class TestEngineEmitsStages:
    async def test_handle_emits_per_stage_stats(self) -> None:
        import json

        spy = _SpySink()
        engine = ProxyEngine(
            upstream=_FakeUpstream(),
            compressor=LosslessCompressor(),
            cache=NullCache(),
            redactor=Redactor(),
            policy=CachePolicy(),
            config=EngineConfig(
                anthropic_upstream="https://a.test",
                openai_upstream="https://o.test",
                cache_enabled=False,
            ),
            metrics=spy,
        )
        body = json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "hi   \n\n\n\nthere"}]}
        ).encode()
        await engine.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        event = spy.events[0]
        lossless = next(s for s in event.stages if s.stage == "lossless")
        assert lossless.ok is True
        assert lossless.tokens_after <= lossless.tokens_before
