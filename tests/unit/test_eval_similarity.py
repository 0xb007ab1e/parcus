"""Tests for the semantic-cache precision gate."""

from __future__ import annotations

from parcus.eval import BUILTIN_SIMILARITY_SAMPLES, SimilaritySample, evaluate_similarity


def test_lexical_embedder_fails_the_adversarial_set() -> None:
    # KEY FINDING: the default dependency-free HashingEmbedder is *lexical* — it cannot
    # distinguish requests that differ only in numbers/entities (extract_terms drops them), so it
    # produces a false hit on the built-in adversarial set even at the high default threshold.
    # The gate correctly FAILS, proving the lexical embedder is unsafe for semantic caching.
    report = evaluate_similarity(BUILTIN_SIMILARITY_SAMPLES)
    assert report.false_hits >= 1
    assert report.passed is False


def test_clean_paraphrase_set_passes() -> None:
    # With only true near-duplicates (and no number/entity traps), there are no false hits.
    samples = (
        SimilaritySample(
            name="identical",
            anchor="render the report",
            variant="render the report",
            should_hit=True,
        ),
        SimilaritySample(
            name="distinct",
            anchor="render the report",
            variant="purge all caches",
            should_hit=False,
        ),
    )
    report = evaluate_similarity(samples)
    assert report.false_hits == 0
    assert report.passed is True


def test_false_hit_fails_the_gate() -> None:
    # A threshold low enough that a non-paraphrase scores as a hit must fail.
    samples = (
        SimilaritySample(
            name="not-a-paraphrase",
            anchor="alpha beta gamma delta",
            variant="alpha beta gamma epsilon",
            should_hit=False,
        ),
    )
    report = evaluate_similarity(samples, threshold=0.1)
    assert report.predicted_hits == 1
    assert report.false_hits == 1
    assert report.passed is False
    assert report.precision == 0.0


def test_recall_counts_caught_paraphrases() -> None:
    samples = (
        SimilaritySample(
            name="dup", anchor="same text here", variant="same text here", should_hit=True
        ),
        SimilaritySample(
            name="miss", anchor="totally different", variant="unrelated words", should_hit=True
        ),
    )
    report = evaluate_similarity(samples, threshold=0.99)
    assert report.true_hits == 1  # the identical pair
    assert 0.0 < report.recall <= 1.0
    assert report.passed is True  # no false hits even though recall < 1


def test_render_contains_verdict() -> None:
    out = evaluate_similarity(BUILTIN_SIMILARITY_SAMPLES).render()
    assert "threshold=" in out
    assert "false_hits=" in out
