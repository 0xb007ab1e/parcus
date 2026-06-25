"""Precision gate for the opt-in semantic (near-duplicate) cache.

A similarity cache serves a cached response when a new request is *near-duplicate* of a prior
one. The danger is a **false hit**: serving the wrong answer because two different requests
scored above the threshold. So the gate that matters is **precision** — of the pairs the cache
*would* treat as the same, how many genuinely are paraphrases. This harness scores a labelled
set of (anchor, variant, should_hit) pairs at a given threshold/embedder and **fails on any
false hit** (precision < 1.0): a paraphrase that should hit is a missed optimization, but a
non-paraphrase that hits is a correctness bug. Mirrors ``parsimony eval --retrieval``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from parsimony.memory.embedding import EmbedderPort, HashingEmbedder, cosine

__all__ = [
    "BUILTIN_SIMILARITY_SAMPLES",
    "SimilarityCase",
    "SimilarityReport",
    "SimilaritySample",
    "evaluate_similarity",
]


@dataclass(frozen=True, slots=True)
class SimilaritySample:
    """A labelled pair.

    Args:
        name: Identifier.
        anchor: A request that is (or would be) cached.
        variant: A later request to test against the anchor.
        should_hit: Whether ``variant`` is a true paraphrase of ``anchor`` (same answer applies).
    """

    name: str
    anchor: str
    variant: str
    should_hit: bool


@dataclass(frozen=True, slots=True)
class SimilarityCase:
    """Per-sample outcome."""

    name: str
    similarity: float
    predicted_hit: bool
    should_hit: bool

    @property
    def correct(self) -> bool:
        """Whether the predicted hit/miss matched the label."""
        return self.predicted_hit == self.should_hit

    @property
    def false_hit(self) -> bool:
        """A non-paraphrase served as a hit — the correctness-critical failure."""
        return self.predicted_hit and not self.should_hit


@dataclass(frozen=True, slots=True)
class SimilarityReport:
    """Aggregate precision/recall report for a threshold + embedder."""

    threshold: float
    cases: tuple[SimilarityCase, ...] = field(default_factory=tuple)

    @property
    def false_hits(self) -> int:
        """Number of non-paraphrases that would be served (must be 0 to pass)."""
        return sum(1 for c in self.cases if c.false_hit)

    @property
    def predicted_hits(self) -> int:
        """Number of pairs the cache would treat as the same."""
        return sum(1 for c in self.cases if c.predicted_hit)

    @property
    def true_hits(self) -> int:
        """Number of correctly-served paraphrases."""
        return sum(1 for c in self.cases if c.predicted_hit and c.should_hit)

    @property
    def precision(self) -> float:
        """Fraction of predicted hits that were true paraphrases (1.0 when none predicted)."""
        return self.true_hits / self.predicted_hits if self.predicted_hits else 1.0

    @property
    def recall(self) -> float:
        """Fraction of true paraphrases that were caught."""
        positives = sum(1 for c in self.cases if c.should_hit)
        return self.true_hits / positives if positives else 1.0

    @property
    def passed(self) -> bool:
        """The gate: no false hits (precision == 1.0)."""
        return self.false_hits == 0

    def render(self) -> str:
        """Render a human-readable summary."""
        lines = [f"{'case':<28} {'sim':>6} {'pred':>5} {'want':>5} {'ok':>4}", "-" * 52]
        for c in self.cases:
            lines.append(
                f"{c.name[:28]:<28} {c.similarity:>6.3f} "
                f"{'hit' if c.predicted_hit else 'miss':>5} "
                f"{'hit' if c.should_hit else 'miss':>5} {'ok' if c.correct else 'XX':>4}"
            )
        lines.append("-" * 52)
        lines.append(
            f"threshold={self.threshold:.3f}  precision={self.precision * 100:.1f}%  "
            f"recall={self.recall * 100:.1f}%  false_hits={self.false_hits}  "
            f"{'PASS' if self.passed else 'FAIL'}"
        )
        return "\n".join(lines)


def evaluate_similarity(
    samples: Iterable[SimilaritySample],
    *,
    threshold: float = 0.97,
    embedder: EmbedderPort | None = None,
) -> SimilarityReport:
    """Score each labelled pair at ``threshold`` and return a precision/recall report.

    Args:
        samples: The labelled (anchor, variant, should_hit) pairs.
        threshold: Cosine threshold treated as a near-duplicate.
        embedder: Local embedder (defaults to the dependency-free :class:`HashingEmbedder`,
            matching the similarity cache's default).

    Returns:
        A :class:`SimilarityReport`. ``passed`` is True only when there are **no false hits**.
    """
    emb = embedder or HashingEmbedder()
    cases: list[SimilarityCase] = []
    for sample in samples:
        anchor_vec, variant_vec = emb.embed([sample.anchor, sample.variant])
        sim = cosine(anchor_vec, variant_vec)
        cases.append(
            SimilarityCase(
                name=sample.name,
                similarity=sim,
                predicted_hit=sim >= threshold,
                should_hit=sample.should_hit,
            )
        )
    return SimilarityReport(threshold=threshold, cases=tuple(cases))


BUILTIN_SIMILARITY_SAMPLES: tuple[SimilaritySample, ...] = (
    SimilaritySample(
        name="paraphrase-whitespace",
        anchor="Summarize the quarterly revenue report.",
        variant="Summarize the quarterly revenue report.   ",
        should_hit=True,
    ),
    SimilaritySample(
        name="paraphrase-trailing-please",
        anchor="List the open security findings.",
        variant="List the open security findings please.",
        should_hit=True,
    ),
    SimilaritySample(
        name="different-intent-shared-words",
        anchor="Delete the production database backup.",
        variant="Restore the production database backup.",
        should_hit=False,
    ),
    SimilaritySample(
        name="different-topic",
        anchor="Explain the caching strategy.",
        variant="Explain the rate limiting strategy.",
        should_hit=False,
    ),
    SimilaritySample(
        name="opposite-number",
        anchor="Scale the service to 10 replicas.",
        variant="Scale the service to 2 replicas.",
        should_hit=False,
    ),
)
