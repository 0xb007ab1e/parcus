"""Tests for the eval runner over the built-in corpus and regression detection."""

from __future__ import annotations

from parcus.eval import BUILTIN_SAMPLES, evaluate
from parcus.model import CanonicalRequest, CompressionStats


class DropMessageCompressor:
    """A broken compressor that drops the last message (a meaning-changing regression)."""

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        trimmed = request.messages[:-1] or request.messages
        return (
            CanonicalRequest(
                dialect=request.dialect,
                model=request.model,
                messages=trimmed,
                system=request.system,
                stream=request.stream,
                tools_json=request.tools_json,
            ),
            (),
        )


def _only(name: str) -> list:
    return [s for s in BUILTIN_SAMPLES if s.name == name]


class TestEvaluate:
    def test_builtin_corpus_saves_tokens_and_passes(self) -> None:
        report = evaluate(BUILTIN_SAMPLES)
        assert report.passed  # lossless: no regressions
        assert report.total_after < report.total_before  # net token savings
        assert report.overall_ratio > 0.0

    def test_passthrough_sample_is_not_canonicalised(self) -> None:
        report = evaluate(BUILTIN_SAMPLES)
        by_name = {r.name: r for r in report.results}
        assert by_name["passthrough-blocks"].canonicalized is False
        assert by_name["passthrough-blocks"].tokens_before == 0

    def test_code_sample_is_equivalent(self) -> None:
        report = evaluate(_only("with-code"))
        assert report.passed  # fenced code preserved byte-for-byte

    def test_regression_is_detected(self) -> None:
        report = evaluate(_only("multi-turn"), compressor=DropMessageCompressor())
        assert not report.passed
        assert len(report.regressions) >= 1

    def test_equivalence_check_can_be_disabled(self) -> None:
        report = evaluate(
            _only("multi-turn"), compressor=DropMessageCompressor(), check_equivalence=False
        )
        assert report.passed  # invariant not enforced for lossy/experimental runs
