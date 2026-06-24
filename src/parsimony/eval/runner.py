"""Run compression over a sample corpus and measure savings + lossless equivalence."""

from __future__ import annotations

from collections.abc import Iterable

from parsimony.compress import LosslessCompressor
from parsimony.eval.dataset import Sample
from parsimony.eval.equivalence import is_lossless_equivalent
from parsimony.eval.metrics import EvalReport, SampleResult
from parsimony.ports import CompressorPort, TokenizerPort
from parsimony.proxy.dialects import parse
from parsimony.tokenize import default_tokenizer

__all__ = ["evaluate"]


def evaluate(
    samples: Iterable[Sample],
    compressor: CompressorPort | None = None,
    tokenizer: TokenizerPort | None = None,
    *,
    check_equivalence: bool = True,
) -> EvalReport:
    """Evaluate ``samples`` through ``compressor`` and return an aggregate report.

    For each sample the request is canonicalised (non-canonicalisable requests are recorded as
    passthrough and contribute no tokens), compressed, measured before/after, and — when
    ``check_equivalence`` is set — checked against the lossless invariant.

    Args:
        samples: The corpus to evaluate.
        compressor: The compressor under test (defaults to the Tier-0 lossless compressor).
        tokenizer: Token counter for measurement (defaults to the heuristic tokenizer).
        check_equivalence: Enforce the lossless no-regression invariant (default True). Set
            False when evaluating a lossy tier that a quality judge will score instead.

    Returns:
        An :class:`EvalReport` aggregating per-sample results.
    """
    comp = compressor or LosslessCompressor()
    tok = tokenizer or default_tokenizer()
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
        equivalent = is_lossless_equivalent(canonical, compressed) if check_equivalence else True
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
