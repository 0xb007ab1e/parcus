"""Tests for span classification (prose mutable, fenced code immutable)."""

from __future__ import annotations

from parsimony.spans import classify_spans


def _reassemble(text: str) -> str:
    return "".join(s.text for s in classify_spans(text))


class TestClassifySpans:
    def test_plain_prose_is_one_mutable_span(self) -> None:
        spans = classify_spans("just prose here")
        assert len(spans) == 1
        assert spans[0].mutable is True

    def test_fenced_code_is_immutable(self) -> None:
        spans = classify_spans("before ```code block``` after")
        assert [s.mutable for s in spans] == [True, False, True]
        assert spans[1].text == "```code block```"

    def test_code_at_start(self) -> None:
        spans = classify_spans("```x``` tail")
        assert [s.mutable for s in spans] == [False, True]

    def test_empty_text_yields_one_empty_mutable_span(self) -> None:
        spans = classify_spans("")
        assert len(spans) == 1
        assert spans[0].text == ""
        assert spans[0].mutable is True

    def test_reassembly_is_lossless(self) -> None:
        for text in ["a", "a ```b``` c ```d``` e", "```only```", ""]:
            assert _reassemble(text) == text
