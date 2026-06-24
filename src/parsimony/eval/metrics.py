"""Per-sample and aggregate token-savings results for the eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["EvalReport", "SampleResult"]


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Measurement for a single evaluated sample.

    Args:
        name: Sample identifier.
        canonicalized: Whether the request was canonicalised (False = passed through untouched).
        tokens_before: Input tokens before compression (0 when not canonicalised).
        tokens_after: Input tokens after compression (0 when not canonicalised).
        equivalent: Whether the lossless invariant held (always True for passthrough).
    """

    name: str
    canonicalized: bool
    tokens_before: int
    tokens_after: int
    equivalent: bool

    @property
    def tokens_saved(self) -> int:
        """Tokens removed for this sample (never negative)."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def ratio(self) -> float:
        """Fraction of input tokens removed, in ``[0.0, 1.0]`` (0 when no input)."""
        if self.tokens_before <= 0:
            return 0.0
        return self.tokens_saved / self.tokens_before


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate report over a set of evaluated samples."""

    results: tuple[SampleResult, ...] = field(default_factory=tuple)

    @property
    def total_before(self) -> int:
        """Total input tokens across canonicalised samples before compression."""
        return sum(r.tokens_before for r in self.results)

    @property
    def total_after(self) -> int:
        """Total input tokens across canonicalised samples after compression."""
        return sum(r.tokens_after for r in self.results)

    @property
    def total_saved(self) -> int:
        """Total tokens saved across all samples."""
        return self.total_before - self.total_after

    @property
    def overall_ratio(self) -> float:
        """Overall fraction of input tokens removed (0 when no input was measured)."""
        if self.total_before <= 0:
            return 0.0
        return self.total_saved / self.total_before

    @property
    def num_canonicalized(self) -> int:
        """Number of samples that were canonicalised (eligible for compression)."""
        return sum(1 for r in self.results if r.canonicalized)

    @property
    def regressions(self) -> tuple[SampleResult, ...]:
        """Samples whose lossless invariant was violated (correctness failures)."""
        return tuple(r for r in self.results if not r.equivalent)

    @property
    def passed(self) -> bool:
        """Whether the eval gate passes (no lossless-equivalence regressions)."""
        return not self.regressions

    def render(self) -> str:
        """Render a human-readable summary table of the report."""
        lines = [
            f"{'sample':<28} {'canon':>5} {'before':>7} {'after':>7} {'saved%':>7} {'ok':>3}",
            "-" * 60,
        ]
        for r in self.results:
            mark = "ok" if r.equivalent else "FAIL"
            canon = "yes" if r.canonicalized else "no"
            lines.append(
                f"{r.name[:28]:<28} {canon:>5} {r.tokens_before:>7} "
                f"{r.tokens_after:>7} {r.ratio * 100:>6.1f}% {mark:>3}"
            )
        lines.append("-" * 60)
        lines.append(
            f"TOTAL  samples={len(self.results)} canonicalised={self.num_canonicalized}  "
            f"tokens {self.total_before}->{self.total_after} "
            f"(saved {self.total_saved}, {self.overall_ratio * 100:.1f}%)  "
            f"regressions={len(self.regressions)}  "
            f"{'PASS' if self.passed else 'FAIL'}"
        )
        return "\n".join(lines)
