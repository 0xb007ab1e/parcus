"""Answer-preservation gate for the *lossy/semantic* compression tiers.

Tier-0 (lossless) and Tier-1 (filler) have model-free invariants, but the **learned** tier
(Tier-2) drops content tokens by design — it has no runtime proof, so its safety is gated
*offline* by judging that the compressed prompt still preserves the information needed to answer
(``std-owasp-llm`` lossy-output risk; ADR 0006). The same harness validates that an **aggressive
filler** set hasn't been mis-tuned to drop real content.

For each labelled sample we compress the prompt through the tier under test and ask a
:class:`~parcus.eval.quality.QualityJudge` whether the result still contains the must-include
facts. The default :class:`~parcus.eval.quality.KeywordRecallJudge` is deterministic and
**CI-safe**; an :class:`~parcus.eval.quality.LLMJudge` can be wired for a stronger offline check.
The compressor is injected, so the gate logic runs in CI with a fake reducer even though the real
LLMLingua model is only available offline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from parcus.eval.quality import KeywordRecallJudge, QualityJudge
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
from parcus.ports import CompressorPort

__all__ = [
    "BUILTIN_JUDGED_SAMPLES",
    "JudgedCase",
    "JudgedReport",
    "JudgedSample",
    "evaluate_judged",
]


@dataclass(frozen=True, slots=True)
class JudgedSample:
    """A labelled prompt whose compressed form must still preserve ``must_include``.

    Args:
        name: Identifier.
        prompt: The (filler-rich) user prompt to compress.
        must_include: Content phrases the compressed prompt must still contain.
    """

    name: str
    prompt: str
    must_include: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgedCase:
    """Per-sample judged result."""

    name: str
    score: float
    passed: bool
    compressed: str


@dataclass(frozen=True, slots=True)
class JudgedReport:
    """Aggregate answer-preservation report (the gate passes only if every case does)."""

    cases: tuple[JudgedCase, ...] = field(default_factory=tuple)

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
        """Whether every case preserved its required information (the gate)."""
        return all(c.passed for c in self.cases)

    def render(self) -> str:
        """Render a human-readable summary."""
        lines = [f"{'case':<28} {'recall':>7} {'ok':>5}", "-" * 44]
        for c in self.cases:
            lines.append(f"{c.name:<28} {c.score:>6.0%} {('ok' if c.passed else 'FAIL'):>5}")
        verdict = "PASS" if self.passed else "FAIL"
        lines.append("-" * 44)
        lines.append(f"{'mean':<28} {self.mean_score:>6.0%} {verdict:>5}")
        return "\n".join(lines)


def _compress_prompt(prompt: str, compressor: CompressorPort) -> str:
    """Compress a single user prompt through ``compressor`` and return the resulting text."""
    request = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="eval",
        messages=(Message(role=Role.USER, spans=(Span(prompt, mutable=True),)),),
    )
    compressed, _ = compressor.compress(request)
    return compressed.messages[0].text


def evaluate_judged(
    samples: Iterable[JudgedSample],
    compressor: CompressorPort,
    judge: QualityJudge | None = None,
) -> JudgedReport:
    """Compress each sample's prompt and judge whether it still preserves the required facts.

    Args:
        samples: The labelled prompts to evaluate.
        compressor: The tier under test (e.g. lossless+aggressive-filler, or +learned). Injected,
            so the gate runs in CI with a fake reducer.
        judge: The quality judge (default :class:`KeywordRecallJudge`, deterministic/CI-safe).

    Returns:
        The aggregate :class:`JudgedReport`.
    """
    judge = judge or KeywordRecallJudge()
    cases: list[JudgedCase] = []
    for sample in samples:
        compressed = _compress_prompt(sample.prompt, compressor)
        verdict = judge.judge(compressed, sample.must_include)
        cases.append(
            JudgedCase(
                name=sample.name,
                score=verdict.score,
                passed=verdict.passed,
                compressed=compressed,
            )
        )
    return JudgedReport(cases=tuple(cases))


# A small corpus: filler-rich prompts whose must-include phrases are content (survive filler
# removal, but a lossy learned tier could drop them — which is exactly what the gate catches).
BUILTIN_JUDGED_SAMPLES: tuple[JudgedSample, ...] = (
    JudgedSample(
        name="scale-replicas",
        prompt="Could you please just go ahead and scale the service to 10 replicas, really soon?",
        must_include=("scale", "10 replicas"),
    ),
    JudgedSample(
        name="caching-tradeoff",
        prompt="Honestly, can you very briefly explain the main trade-off of response caching?",
        must_include=("trade-off", "caching"),
    ),
    JudgedSample(
        name="timeout-budget",
        prompt="Please just set the timeout to 30 seconds and keep the budget cap at 500 dollars.",
        must_include=("timeout", "30 seconds", "500 dollars"),
    ),
    JudgedSample(
        name="define-idempotent",
        prompt="I'd really appreciate it if you could simply define what idempotent means.",
        must_include=("idempotent", "define"),
    ),
)
