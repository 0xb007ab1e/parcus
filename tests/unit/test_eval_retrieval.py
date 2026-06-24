"""Tests for the retrieval-quality gate."""

from __future__ import annotations

from parsimony.eval.retrieval import (
    BUILTIN_RETRIEVAL_SAMPLES,
    RetrievalReport,
    RetrievalSample,
    evaluate_retrieval,
)


class TestEvaluateRetrieval:
    def test_builtin_samples_pass_the_gate(self) -> None:
        report = evaluate_retrieval(BUILTIN_RETRIEVAL_SAMPLES)
        assert report.passed  # lexical retrieval recalls the labelled facts on these cases
        assert report.mean_score == 1.0
        assert report.num_passed == len(BUILTIN_RETRIEVAL_SAMPLES)

    def test_detects_a_recall_miss(self) -> None:
        # The needed fact is present in context but the query shares no terms with it, so
        # lexical retrieval misses it -> the gate fails (exactly what should block wiring).
        sample = RetrievalSample(
            name="lexical-miss",
            context=("Passwords are hashed with argon2id.", "unrelated filler text here"),
            query="cryptographic credential storage approach",
            must_include=("argon2id",),
        )
        report = evaluate_retrieval([sample])
        assert not report.passed
        assert report.cases[0].score == 0.0

    def test_render_and_empty_report(self) -> None:
        rendered = evaluate_retrieval(BUILTIN_RETRIEVAL_SAMPLES).render()
        assert "TOTAL" in rendered
        assert "PASS" in rendered
        assert RetrievalReport().mean_score == 0.0
        assert RetrievalReport().passed is True  # vacuously (no failing cases)
