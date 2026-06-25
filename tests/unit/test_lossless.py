"""Tests for the Tier-0 lossless compressor and whitespace normalisation."""

from __future__ import annotations

from parcus.compress.lossless import (
    LosslessCompressor,
    normalise_code_aware,
    normalise_whitespace,
)
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*spans: Span, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="claude-test",
        messages=(Message(role=Role.USER, spans=spans),),
        system=system,
    )


class TestNormaliseWhitespace:
    def test_strips_trailing_whitespace(self) -> None:
        assert normalise_whitespace("a   \nb\t\nc") == "a\nb\nc"

    def test_collapses_excess_blank_lines(self) -> None:
        assert normalise_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_collapses_blank_line_runs_including_edges(self) -> None:
        # Boundary blank-line runs are collapsed (to one blank line), not fully removed, so
        # prose adjacent to a code span keeps its separation.
        assert normalise_whitespace("\n\n\n\nhello\n\n\n\n") == "\n\nhello\n\n"

    def test_preserves_interior_spacing_and_indentation(self) -> None:
        # Single interior spaces and leading indentation are meaning-bearing; keep them.
        assert normalise_whitespace("def f():\n    return 1") == "def f():\n    return 1"

    def test_idempotent(self) -> None:
        once = normalise_whitespace("a   \n\n\n\nb  ")
        assert normalise_whitespace(once) == once


class TestNormaliseCodeAware:
    def test_normalises_prose_outside_code(self) -> None:
        assert normalise_code_aware("hi   \n\n\n\nbye   ") == "hi\n\nbye"

    def test_preserves_fenced_code_verbatim(self) -> None:
        text = "intro   \n\n\n\n```x = 1   \ny = 2   ```"
        out = normalise_code_aware(text)
        assert "```x = 1   \ny = 2   ```" in out  # code byte-for-byte
        assert out.startswith("intro\n\n")


class TestLosslessCompressor:
    def test_reduces_tokens_on_mutable_prose(self) -> None:
        before = "Please do the thing.    \n\n\n\n\nThanks.   "
        req = _req(Span(before, mutable=True))
        out, stats = LosslessCompressor().compress(req)
        assert len(stats) == 1
        assert stats[0].step == "lossless"
        assert stats[0].spans_touched == 1
        assert stats[0].tokens_after <= stats[0].tokens_before
        assert stats[0].tokens_saved >= 0
        assert out.messages[0].spans[0].text == "Please do the thing.\n\nThanks."

    def test_never_alters_immutable_spans(self) -> None:
        code = "x = 1   \n\n\n\n\ny = 2   "  # trailing ws + blank lines, but immutable
        req = _req(Span(code, mutable=False))
        out, stats = LosslessCompressor().compress(req)
        assert out.messages[0].spans[0].text == code  # byte-for-byte preserved
        assert stats[0].spans_touched == 0

    def test_mixed_spans_only_touch_mutable(self) -> None:
        req = _req(
            Span("Here is code:\n\n\n\n", mutable=True),
            Span("def f():\n    pass   ", mutable=False),
        )
        out, stats = LosslessCompressor().compress(req)
        # Boundary blank-line run collapsed to one blank line (not removed) — keeps the
        # separation from the following immutable code span.
        assert out.messages[0].spans[0].text == "Here is code:\n\n"
        assert out.messages[0].spans[1].text == "def f():\n    pass   "
        assert stats[0].spans_touched == 1

    def test_no_change_when_already_clean(self) -> None:
        req = _req(Span("already clean", mutable=True))
        out, stats = LosslessCompressor().compress(req)
        assert out.messages[0].spans[0].text == "already clean"
        assert stats[0].spans_touched == 0
        assert stats[0].tokens_saved == 0

    def test_compresses_the_system_prompt(self) -> None:
        req = _req(Span("hi", mutable=True), system="be terse   \n\n\n\n\nalways   ")
        out, stats = LosslessCompressor().compress(req)
        assert out.system == "be terse\n\nalways"
        assert stats[0].spans_touched >= 1

    def test_clean_system_is_unchanged(self) -> None:
        req = _req(Span("clean", mutable=True), system="already clean")
        out, stats = LosslessCompressor().compress(req)
        assert out.system == "already clean"
        assert stats[0].spans_touched == 0

    def test_fails_open_on_tokenizer_error(self) -> None:
        class _Boom:
            def count(self, text: str, model: str | None = None) -> int:
                raise RuntimeError("boom")

        req = _req(Span("anything   \n\n\n\n", mutable=True))
        out, stats = LosslessCompressor(tokenizer=_Boom()).compress(req)
        # Fail open: original request returned unchanged, no stats, no exception.
        assert out is req
        assert stats == ()
