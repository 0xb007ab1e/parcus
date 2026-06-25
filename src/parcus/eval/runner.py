"""Run compression over a sample corpus and measure savings + lossless equivalence."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from parcus.compress import LosslessCompressor
from parcus.eval.dataset import Sample
from parcus.eval.equivalence import is_lossless_equivalent
from parcus.eval.metrics import EvalReport, SampleResult
from parcus.model import CanonicalRequest
from parcus.ports import CompressorPort, TokenizerPort
from parcus.proxy.dialects import parse
from parcus.tokenize import default_tokenizer

__all__ = ["evaluate"]

EquivalenceCheck = Callable[[CanonicalRequest, CanonicalRequest], bool]


def evaluate(
    samples: Iterable[Sample],
    compressor: CompressorPort | None = None,
    tokenizer: TokenizerPort | None = None,
    *,
    check_equivalence: bool = True,
    equivalence: EquivalenceCheck | None = None,
) -> EvalReport:
    """Evaluate ``samples`` through ``compressor`` and return an aggregate report.

    For each sample the request is canonicalised (non-canonicalisable requests are recorded as
    passthrough and contribute no tokens), compressed, measured before/after, and — when
    ``check_equivalence`` is set — checked against the lossless invariant.

    Args:
        samples: The corpus to evaluate.
        compressor: The compressor under test (defaults to the Tier-0 lossless compressor).
        tokenizer: Token counter for measurement (defaults to the heuristic tokenizer).
        check_equivalence: Enforce a no-regression invariant (default True). Set False when a
            quality judge will score the output instead.
        equivalence: The invariant to enforce when ``check_equivalence`` is set (default the
            lossless invariant). For the filler tier, pass an :func:`is_filler_equivalent`
            closure bound to the allowed filler set.

    Returns:
        An :class:`EvalReport` aggregating per-sample results.
    """
    comp = compressor or LosslessCompressor()
    tok = tokenizer or default_tokenizer()
    gate = equivalence or is_lossless_equivalent
    results: list[SampleResult] = []

    for sample in samples:
        canonical = parse(sample.dialect, sample.body)
        if canonical is None:
            results.append(
                SampleResult(
                    name=sample.name,
                    canonicalized=False,
                    tokens_before=0,
                    tokens_after=0,
                    equivalent=True,
                )
            )
            continue
        before = tok.count(canonical.text, canonical.model)
        compressed, _stats = comp.compress(canonical)
        after = tok.count(compressed.text, compressed.model)
        equivalent = gate(canonical, compressed) if check_equivalence else True
        results.append(
            SampleResult(
                name=sample.name,
                canonicalized=True,
                tokens_before=before,
                tokens_after=after,
                equivalent=equivalent,
            )
        )

    return EvalReport(tuple(results))
