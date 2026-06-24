"""Retrieval-quality gate for Track B (context retrieval).

Before memory is allowed to *replace* re-sent context in the live path, retrieval must be shown
to **preserve the information needed to answer**. For each labelled sample we ingest the prior
context into a fresh memory, retrieve for the query, and judge whether the retrieved snippets
contain the must-include facts (model-free recall by default). This is the gate the wired
Track B slice will have to pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from parsimony.eval.quality import KeywordRecallJudge, QualityJudge
from parsimony.memory import GraphMemory
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span

__all__ = [
    "BUILTIN_RETRIEVAL_SAMPLES",
    "RetrievalCase",
    "RetrievalReport",
    "RetrievalSample",
    "evaluate_retrieval",
]


@dataclass(frozen=True, slots=True)
class RetrievalSample:
    """A labelled retrieval case.

    Args:
        name: Identifier.
        context: Prior snippets to ingest into memory.
        query: The query to retrieve for.
        must_include: Phrases the retrieved context must contain to count as preserving info.
    """

    name: str
    context: tuple[str, ...]
    query: str
    must_include: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """Per-sample retrieval result."""

    name: str
    score: float
    passed: bool
    retrieved: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Aggregate retrieval-quality report."""

    cases: tuple[RetrievalCase, ...] = field(default_factory=tuple)

    @property
    def mean_score(self) -> float:
        """Mean recall score across cases (0 when empty)."""
        return sum(c.score for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def num_passed(self) -> int:
        """Number of cases that met the quality bar."""
        return sum(1 for c in self.cases if c.passed)

    @property
    def passed(self) -> bool:
        """Whether every case passed (the gate)."""
        return all(c.passed for c in self.cases)

    def render(self) -> str:
        """Render a human-readable summary."""
        lines = [f"{'case':<28} {'recall':>7} {'ok':>5}", "-" * 44]
        for c in self.cases:
            lines.append(
                f"{c.name[:28]:<28} {c.score * 100:>6.1f}% {'ok' if c.passed else 'FAIL':>5}"
            )
        lines.append("-" * 44)
        lines.append(
            f"TOTAL cases={len(self.cases)} passed={self.num_passed} "
            f"mean_recall={self.mean_score * 100:.1f}%  {'PASS' if self.passed else 'FAIL'}"
        )
        return "\n".join(lines)


def _context_request(context: tuple[str, ...]) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model=None,
        messages=tuple(Message(role=Role.USER, spans=(Span(text),)) for text in context),
    )


def evaluate_retrieval(
    samples: Iterable[RetrievalSample],
    *,
    limit: int = 5,
    judge: QualityJudge | None = None,
) -> RetrievalReport:
    """Evaluate retrieval recall for each sample and return an aggregate report.

    Args:
        samples: The labelled retrieval cases.
        limit: Max snippets to retrieve per query.
        judge: Quality judge (defaults to a strict keyword-recall judge).

    Returns:
        A :class:`RetrievalReport`.
    """
    quality = judge or KeywordRecallJudge()
    cases: list[RetrievalCase] = []
    for sample in samples:
        memory = GraphMemory()
        memory.ingest(_context_request(sample.context))
        retrieved = memory.relevant(sample.query, limit=limit)
        verdict = quality.judge(" ".join(retrieved), sample.must_include)
        cases.append(
            RetrievalCase(
                name=sample.name,
                score=verdict.score,
                passed=verdict.passed,
                retrieved=retrieved,
            )
        )
    return RetrievalReport(tuple(cases))


BUILTIN_RETRIEVAL_SAMPLES: tuple[RetrievalSample, ...] = (
    RetrievalSample(
        name="schema-decision",
        context=(
            "The database migration adds a tenant_id column with row-level security.",
            "Standup is at 9am daily.",
            "The response cache uses an exact-hash key.",
        ),
        query="tenant_id row-level security migration",
        must_include=("tenant_id", "row-level security"),
    ),
    RetrievalSample(
        name="cache-design",
        context=(
            "The response cache uses an exact-hash key and never stores prompts.",
            "Lunch is catered on fridays.",
            "The proxy binds loopback only.",
        ),
        query="exact-hash cache key prompts stored",
        must_include=("exact-hash", "never stores prompts"),
    ),
    RetrievalSample(
        name="bind-policy",
        context=(
            "The proxy binds loopback and tailnet, never public.",
            "The coffee machine is broken again.",
        ),
        query="proxy binds loopback tailnet public",
        must_include=("loopback", "tailnet"),
    ),
)
