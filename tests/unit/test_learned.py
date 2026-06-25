"""Tests for the Tier-2 learned compressor (span handling via a fake local reducer)."""

from __future__ import annotations

import math

from parcus.compress.learned import LearnedCompressor
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*spans: Span, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=spans),),
        system=system,
    )


class _HalfReducer:
    """Deterministic stand-in: keep the first ``ceil(keep_ratio * n)`` whitespace tokens."""

    def reduce(self, text: str, *, keep_ratio: float) -> str:
        words = text.split()
        if not words:
            return text
        keep = max(1, math.ceil(len(words) * keep_ratio))
        return " ".join(words[:keep])


class _BoomReducer:
    def reduce(self, text: str, *, keep_ratio: float) -> str:
        raise RuntimeError("model exploded")


class _NoopReducer:
    def reduce(self, text: str, *, keep_ratio: float) -> str:
        return text


class TestLearnedCompressor:
    def test_reduces_mutable_prose(self) -> None:
        comp = LearnedCompressor(_HalfReducer(), keep_ratio=0.5)
        out, stats = comp.compress(_req(Span("one two three four five six")))
        assert out.messages[0].text == "one two three"  # ceil(6*0.5)=3 kept
        assert stats[0].step == "learned"
        assert stats[0].spans_touched == 1
        assert stats[0].tokens_after <= stats[0].tokens_before

    def test_reports_no_runtime_invariant(self) -> None:
        # Tier-2 has no model-free invariant -> ok is None (accuracy comes from the offline gate).
        _out, stats = LearnedCompressor(_HalfReducer()).compress(_req(Span("a b c d")))
        assert stats[0].ok is None

    def test_never_touches_immutable_spans(self) -> None:
        comp = LearnedCompressor(_HalfReducer(), keep_ratio=0.5)
        out, stats = comp.compress(_req(Span("alpha beta gamma delta", mutable=False)))
        assert out.messages[0].spans[0].text == "alpha beta gamma delta"
        assert stats[0].spans_touched == 0

    def test_reduces_system_prose_but_preserves_code(self) -> None:
        comp = LearnedCompressor(_HalfReducer(), keep_ratio=0.5)
        out, _stats = comp.compress(_req(Span("hi"), system="aa bb cc dd ```keep all code here```"))
        assert "```keep all code here```" in (out.system or "")  # fenced code untouched
        assert "aa bb" in (out.system or "")  # prose reduced

    def test_noop_reducer_leaves_request_unchanged(self) -> None:
        out, stats = LearnedCompressor(_NoopReducer()).compress(
            _req(Span("nothing to cut"), system="also unchanged")
        )
        assert out.messages[0].text == "nothing to cut"
        assert out.system == "also unchanged"  # system present but reduce is a no-op
        assert stats[0].spans_touched == 0

    def test_fails_open_when_reducer_raises(self) -> None:
        request = _req(Span("this will explode"))
        out, stats = LearnedCompressor(_BoomReducer()).compress(request)
        assert out is request  # original returned unchanged
        assert stats == ()  # no stats on the fail-open path
