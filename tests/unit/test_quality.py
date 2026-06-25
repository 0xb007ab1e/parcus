"""Tests for the quality judges."""

from __future__ import annotations

from parcus.eval.quality import KeywordRecallJudge, LLMJudge, QualityJudge


class TestKeywordRecallJudge:
    def test_all_present_passes(self) -> None:
        v = KeywordRecallJudge().judge(
            "uses argon2id and row-level security", ("argon2id", "row-level")
        )
        assert v.passed is True
        assert v.score == 1.0

    def test_partial_fails_at_default_threshold(self) -> None:
        v = KeywordRecallJudge().judge("uses argon2id only", ("argon2id", "row-level"))
        assert v.passed is False
        assert v.score == 0.5
        assert "row-level" in v.reason

    def test_partial_passes_below_threshold(self) -> None:
        v = KeywordRecallJudge(threshold=0.5).judge("argon2id", ("argon2id", "missing"))
        assert v.passed is True
        assert v.score == 0.5

    def test_no_requirements_passes(self) -> None:
        v = KeywordRecallJudge().judge("anything", ())
        assert v.passed is True
        assert v.score == 1.0

    def test_is_case_insensitive(self) -> None:
        assert KeywordRecallJudge().judge("ARGON2ID", ("argon2id",)).passed is True

    def test_satisfies_protocol(self) -> None:
        assert isinstance(KeywordRecallJudge(), QualityJudge)


class TestLLMJudge:
    def test_yes_reply_passes(self) -> None:
        v = LLMJudge(lambda _prompt: "Yes, all facts present.").judge("text", ("fact",))
        assert v.passed is True
        assert v.score == 1.0

    def test_no_reply_fails(self) -> None:
        v = LLMJudge(lambda _prompt: "No, missing one.").judge("text", ("fact",))
        assert v.passed is False
        assert v.score == 0.0

    def test_satisfies_protocol(self) -> None:
        assert isinstance(LLMJudge(lambda _p: "yes"), QualityJudge)
