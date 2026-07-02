"""Tests for the answer-preservation gate (lossy-tier judged eval)."""

from __future__ import annotations

from parcus.compress import (
    AGGRESSIVE_FILLERS,
    ChainCompressor,
    DedupCompressor,
    FillerCompressor,
    LearnedCompressor,
    LosslessCompressor,
    NullCompressor,
    ToolResultElider,
)
from parcus.eval import (
    BUILTIN_DEDUP_SAMPLES,
    BUILTIN_ELISION_SAMPLES,
    BUILTIN_JUDGED_SAMPLES,
    JudgedDedupSample,
    JudgedElisionSample,
    JudgedSample,
    evaluate_judged,
    evaluate_judged_dedup,
    evaluate_judged_elision,
)
from parcus.eval.quality import KeywordRecallJudge
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span


class _DropReducer:
    """A fake learned reducer that simulates a lossy drop of a target phrase (no model needed)."""

    def __init__(self, drop: str) -> None:
        self._drop = drop

    def reduce(self, text: str, *, keep_ratio: float) -> str:
        return text.replace(self._drop, "")


_SAMPLES = (
    JudgedSample(name="keeps", prompt="please scale to 10 replicas", must_include=("10 replicas",)),
    JudgedSample(name="empty", prompt="anything", must_include=()),
)


def _elision_req(*messages: Message) -> CanonicalRequest:
    return CanonicalRequest(dialect=Dialect.ANTHROPIC, model="eval", messages=tuple(messages))


def _raw_tool_result(text: str) -> Message:
    block = {"type": "tool_result", "tool_use_id": "t", "content": text}
    return Message(role=Role.USER, spans=(), raw={"role": "user", "content": [block]})


class TestJudgedElision:
    def test_gate_passes_on_builtin_corpus(self) -> None:
        report = evaluate_judged_elision(BUILTIN_ELISION_SAMPLES, ToolResultElider(keep_recent=4))
        assert report.passed
        assert report.mean_score == 1.0

    def test_gate_fails_when_stale_not_removed(self) -> None:
        # keep_recent covers everything → nothing is elided → the must_drop phrase survives → FAIL.
        report = evaluate_judged_elision(BUILTIN_ELISION_SAMPLES, ToolResultElider(keep_recent=100))
        assert not report.passed

    def test_gate_fails_on_over_elision(self) -> None:
        # The answer-relevant phrase lives *inside* the (large) tool result that gets elided →
        # recall drops → the gate correctly flags the loss.
        sample = JudgedElisionSample(
            name="over-elide",
            request=_elision_req(
                _raw_tool_result("CRITICAL ANSWER: 42 widgets. " + "padding text; " * 50)
            ),
            must_include=("42 widgets",),
            must_drop=(),
        )
        report = evaluate_judged_elision((sample,), ToolResultElider(keep_recent=0))
        assert not report.passed


def _msg(role: Role, *spans: Span) -> Message:
    return Message(role=role, spans=spans)


class TestJudgedDedup:
    _BIG = "10 replicas and a 30 second timeout. " + "padding detail line; " * 30

    def test_gate_passes_on_builtin_corpus(self) -> None:
        report = evaluate_judged_dedup(BUILTIN_DEDUP_SAMPLES, DedupCompressor())
        assert report.passed
        assert report.mean_score == 1.0

    def test_gate_fails_when_nothing_deduped(self) -> None:
        # A large block that appears only once → dedup doesn't fire → the gate must NOT pass
        # vacuously even though the content is trivially present.
        sample = JudgedDedupSample(
            name="no-repeat",
            request=_elision_req(_msg(Role.USER, Span(self._BIG))),
            must_include=("10 replicas",),
        )
        report = evaluate_judged_dedup((sample,), DedupCompressor())
        assert not report.passed

    def test_gate_fails_when_content_missing(self) -> None:
        # A repeated block is deduped, but the required phrase isn't present → recall fails.
        sample = JudgedDedupSample(
            name="missing",
            request=_elision_req(
                _msg(Role.USER, Span(self._BIG)), _msg(Role.USER, Span(self._BIG))
            ),
            must_include=("nonexistent phrase",),
        )
        report = evaluate_judged_dedup((sample,), DedupCompressor())
        assert not report.passed


class TestEvaluateJudged:
    def test_identity_compressor_preserves_everything(self) -> None:
        report = evaluate_judged(_SAMPLES, NullCompressor())
        assert report.passed
        assert report.num_passed == 2
        assert report.mean_score == 1.0

    def test_empty_requirements_pass(self) -> None:
        report = evaluate_judged(
            (JudgedSample(name="x", prompt="hi", must_include=()),), NullCompressor()
        )
        assert report.passed

    def test_aggressive_filler_preserves_required_content(self) -> None:
        # The actual offline validation: the AGGRESSIVE_FILLERS set must not drop any required
        # content phrase from the built-in corpus (it only removes discourse fillers).
        compressor = ChainCompressor(
            [LosslessCompressor(), FillerCompressor(fillers=AGGRESSIVE_FILLERS)]
        )
        report = evaluate_judged(BUILTIN_JUDGED_SAMPLES, compressor, KeywordRecallJudge())
        assert report.passed, report.render()

    def test_learned_gate_catches_dropped_content(self) -> None:
        # The CI-able proof of the learned gate: a lossy reducer that drops "10 replicas" must make
        # the scale-replicas sample fail — no real model required (the reducer seam is faked).
        compressor = ChainCompressor(
            [LosslessCompressor(), LearnedCompressor(_DropReducer("10 replicas"), keep_ratio=0.5)]
        )
        report = evaluate_judged(BUILTIN_JUDGED_SAMPLES, compressor, KeywordRecallJudge())
        assert not report.passed
        failed = [c.name for c in report.cases if not c.passed]
        assert "scale-replicas" in failed

    def test_report_render_marks_failures(self) -> None:
        compressor = ChainCompressor(
            [LosslessCompressor(), LearnedCompressor(_DropReducer("10 replicas"), keep_ratio=0.5)]
        )
        report = evaluate_judged(BUILTIN_JUDGED_SAMPLES, compressor, KeywordRecallJudge())
        rendered = report.render()
        assert "FAIL" in rendered
        assert "scale-replicas" in rendered

    def test_partial_recall_scores_between_zero_and_one(self) -> None:
        sample = JudgedSample(name="two", prompt="alpha beta", must_include=("alpha", "gamma"))
        report = evaluate_judged((sample,), NullCompressor())
        assert report.cases[0].score == 0.5  # 1 of 2 present
        assert not report.passed
