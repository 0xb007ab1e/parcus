"""Tests for the Tier-1 filler-removal compressor."""

from __future__ import annotations

from parsimony.compress.filler import (
    AGGRESSIVE_FILLERS,
    DEFAULT_FILLERS,
    FillerCompressor,
    strip_fillers,
)
from parsimony.eval import is_filler_equivalent
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*spans: Span, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=spans),),
        system=system,
    )


class TestStripFillers:
    def test_removes_default_fillers(self) -> None:
        assert strip_fillers("please just fix this") == "fix this"

    def test_keeps_non_filler_words(self) -> None:
        assert strip_fillers("keep all of these words") == "keep all of these words"

    def test_preserves_fenced_code(self) -> None:
        out = strip_fillers("please run ```just do x```")
        assert "```just do x```" in out  # 'just' inside code is NOT removed
        assert "please" not in out

    def test_honours_custom_filler_set(self) -> None:
        assert strip_fillers("foo bar baz", frozenset({"foo", "baz"})) == "bar"

    def test_default_set_membership(self) -> None:
        assert "please" in DEFAULT_FILLERS
        assert "fix" not in DEFAULT_FILLERS


class TestFillerCompressor:
    def test_removes_fillers_from_messages(self) -> None:
        out, stats = FillerCompressor().compress(_req(Span("please fix really now")))
        assert out.messages[0].text == "fix now"
        assert stats[0].step == "filler"
        assert stats[0].spans_touched == 1
        assert stats[0].tokens_after <= stats[0].tokens_before

    def test_never_touches_immutable_spans(self) -> None:
        out, stats = FillerCompressor().compress(_req(Span("please x", mutable=False)))
        assert out.messages[0].spans[0].text == "please x"
        assert stats[0].spans_touched == 0

    def test_compresses_the_system_prompt(self) -> None:
        out, stats = FillerCompressor().compress(_req(Span("hi"), system="please be very terse"))
        assert out.system == "be terse"
        assert stats[0].spans_touched >= 1

    def test_no_fillers_leaves_touched_zero(self) -> None:
        out, stats = FillerCompressor().compress(_req(Span("fix the bug")))
        assert out.messages[0].text == "fix the bug"
        assert stats[0].spans_touched == 0

    def test_fails_open_on_tokenizer_error(self) -> None:
        class _Boom:
            def count(self, text: str, model: str | None = None) -> int:
                raise RuntimeError("boom")

        req = _req(Span("please just fix"))
        out, stats = FillerCompressor(tokenizer=_Boom()).compress(req)
        assert out is req
        assert stats == ()


class TestAggressiveFillers:
    def test_is_a_superset_of_default(self) -> None:
        assert DEFAULT_FILLERS <= AGGRESSIVE_FILLERS
        assert len(AGGRESSIVE_FILLERS) > len(DEFAULT_FILLERS)

    def test_removes_more_than_default(self) -> None:
        text = "this is obviously and clearly just a test"
        default_out = strip_fillers(text, DEFAULT_FILLERS)
        aggressive_out = strip_fillers(text, AGGRESSIVE_FILLERS)
        assert "obviously" in default_out and "clearly" in default_out  # not in the small set
        assert "obviously" not in aggressive_out and "clearly" not in aggressive_out
        assert "just" not in aggressive_out  # shared with default

    def test_aggressive_removal_still_passes_model_free_guardrail(self) -> None:
        # The structural invariant holds for ANY allow-list: only listed whole tokens are removed.
        req = CanonicalRequest(
            dialect=Dialect.ANTHROPIC,
            model="m",
            messages=(Message(role=Role.USER, spans=(Span("obviously fix this clearly now"),)),),
        )
        out, _stats = FillerCompressor(fillers=AGGRESSIVE_FILLERS).compress(req)
        assert is_filler_equivalent(req, out, AGGRESSIVE_FILLERS)
        assert out.messages[0].text == "fix this now"
