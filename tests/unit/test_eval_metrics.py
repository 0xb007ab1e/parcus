"""Tests for the eval metrics aggregation and report rendering."""

from __future__ import annotations

from parcus.eval.metrics import EvalReport, SampleResult


class TestSampleResult:
    def test_saved_and_ratio(self) -> None:
        r = SampleResult(
            "x", canonicalized=True, tokens_before=100, tokens_after=60, equivalent=True
        )
        assert r.tokens_saved == 40
        assert r.ratio == 0.4

    def test_ratio_zero_without_input(self) -> None:
        r = SampleResult("y", canonicalized=False, tokens_before=0, tokens_after=0, equivalent=True)
        assert r.ratio == 0.0


class TestEvalReport:
    def test_aggregates_and_renders_pass(self) -> None:
        report = EvalReport(
            (
                SampleResult(
                    "a", canonicalized=True, tokens_before=100, tokens_after=60, equivalent=True
                ),
                SampleResult(
                    "b", canonicalized=False, tokens_before=0, tokens_after=0, equivalent=True
                ),
            )
        )
        assert report.total_before == 100
        assert report.total_after == 60
        assert report.total_saved == 40
        assert report.num_canonicalized == 1
        assert report.passed is True
        rendered = report.render()
        assert "TOTAL" in rendered
        assert "PASS" in rendered

    def test_detects_regression_in_render(self) -> None:
        report = EvalReport(
            (
                SampleResult(
                    "a", canonicalized=True, tokens_before=10, tokens_after=9, equivalent=False
                ),
            )
        )
        assert report.passed is False
        assert "FAIL" in report.render()

    def test_overall_ratio_zero_when_empty(self) -> None:
        assert EvalReport(()).overall_ratio == 0.0
