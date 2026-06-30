"""Tests for the answer-preservation gate (lossy-tier judged eval)."""

from __future__ import annotations

from parcus.compress import (
    AGGRESSIVE_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LearnedCompressor,
    LosslessCompressor,
    NullCompressor,
)
from parcus.eval import BUILTIN_JUDGED_SAMPLES, JudgedSample, evaluate_judged
from parcus.eval.quality import KeywordRecallJudge


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
