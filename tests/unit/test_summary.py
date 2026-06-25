"""Tests for the Track C summarisers."""

from __future__ import annotations

from parcus.memory.summary import ExtractiveSummarizer, LLMSummarizer, Summarizer


class TestExtractiveSummarizer:
    def test_distinct_leads_in_order(self) -> None:
        out = ExtractiveSummarizer().summarize(
            ["First turn. extra detail.", "Second turn here", "First turn. extra detail."]
        )
        lines = out.splitlines()
        assert lines == ["- First turn.", "- Second turn here"]  # deduped, order preserved

    def test_first_sentence_only(self) -> None:
        assert ExtractiveSummarizer().summarize(["Keep this. Drop that."]) == "- Keep this."

    def test_respects_max_items(self) -> None:
        out = ExtractiveSummarizer().summarize(["a one", "b two", "c three"], max_items=2)
        assert len(out.splitlines()) == 2

    def test_empty_input_is_empty(self) -> None:
        assert ExtractiveSummarizer().summarize([]) == ""
        assert ExtractiveSummarizer().summarize(["", "   "]) == ""

    def test_satisfies_protocol(self) -> None:
        assert isinstance(ExtractiveSummarizer(), Summarizer)


class TestLLMSummarizer:
    def test_delegates_to_completion(self) -> None:
        out = LLMSummarizer(lambda _prompt: "  - summarised  ").summarize(["x", "y"])
        assert out == "- summarised"

    def test_satisfies_protocol(self) -> None:
        assert isinstance(LLMSummarizer(lambda _p: ""), Summarizer)
