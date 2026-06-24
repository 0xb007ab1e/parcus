"""Tests for the canonical model and compression statistics."""

from __future__ import annotations

from parsimony.model import (
    CanonicalRequest,
    CompressionStats,
    Dialect,
    Message,
    Role,
    Span,
)


class TestSpan:
    def test_with_text_preserves_mutable_flag(self) -> None:
        span = Span("old", mutable=False)
        new = span.with_text("new")
        assert new.text == "new"
        assert new.mutable is False
        assert span.text == "old"  # original unchanged (frozen)


class TestMessageAndRequestText:
    def test_message_text_concatenates_spans(self) -> None:
        msg = Message(role=Role.USER, spans=(Span("ab"), Span("cd")))
        assert msg.text == "abcd"

    def test_request_text_includes_system(self) -> None:
        req = CanonicalRequest(
            dialect=Dialect.OPENAI,
            model="gpt-test",
            messages=(Message(role=Role.USER, spans=(Span("hi"),)),),
            system="sys",
        )
        assert req.text == "syshi"


class TestCompressionStats:
    def test_tokens_saved_and_ratio(self) -> None:
        stats = CompressionStats(step="p", tokens_before=100, tokens_after=75)
        assert stats.tokens_saved == 25
        assert stats.ratio == 0.25

    def test_ratio_zero_when_no_input(self) -> None:
        stats = CompressionStats(step="p", tokens_before=0, tokens_after=0)
        assert stats.ratio == 0.0

    def test_tokens_saved_clamped_at_zero(self) -> None:
        stats = CompressionStats(step="p", tokens_before=10, tokens_after=20)
        assert stats.tokens_saved == 0
