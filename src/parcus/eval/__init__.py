"""Token-savings measurement harness with a lossless no-regression gate.

The headline metric — **% input-token reduction at equal quality** — is produced here.
For the always-on lossless tier, "equal quality" is enforced model-free by
:mod:`parcus.eval.equivalence`; lossy tiers will plug a quality judge into
:func:`parcus.eval.runner.evaluate` (``check_equivalence=False``) in a later milestone.
"""

from parcus.eval.dataset import Sample, load_jsonl
from parcus.eval.equivalence import (
    filler_violations,
    is_filler_equivalent,
    is_lossless_equivalent,
    lossless_violations,
)
from parcus.eval.judged import (
    BUILTIN_ELISION_SAMPLES,
    BUILTIN_JUDGED_SAMPLES,
    JudgedElisionSample,
    JudgedReport,
    JudgedSample,
    evaluate_judged,
    evaluate_judged_elision,
)
from parcus.eval.metrics import EvalReport, SampleResult
from parcus.eval.quality import KeywordRecallJudge, LLMJudge, QualityJudge, QualityVerdict
from parcus.eval.retrieval import (
    BUILTIN_RETRIEVAL_SAMPLES,
    RetrievalReport,
    RetrievalSample,
    evaluate_retrieval,
)
from parcus.eval.runner import evaluate
from parcus.eval.samples import BUILTIN_SAMPLES
from parcus.eval.similarity import (
    BUILTIN_SIMILARITY_SAMPLES,
    SimilarityReport,
    SimilaritySample,
    evaluate_similarity,
)

__all__ = [
    "BUILTIN_ELISION_SAMPLES",
    "BUILTIN_JUDGED_SAMPLES",
    "BUILTIN_RETRIEVAL_SAMPLES",
    "BUILTIN_SAMPLES",
    "BUILTIN_SIMILARITY_SAMPLES",
    "EvalReport",
    "JudgedElisionSample",
    "JudgedReport",
    "JudgedSample",
    "KeywordRecallJudge",
    "LLMJudge",
    "QualityJudge",
    "QualityVerdict",
    "RetrievalReport",
    "RetrievalSample",
    "Sample",
    "SampleResult",
    "SimilarityReport",
    "SimilaritySample",
    "evaluate",
    "evaluate_judged",
    "evaluate_judged_elision",
    "evaluate_retrieval",
    "evaluate_similarity",
    "filler_violations",
    "is_filler_equivalent",
    "is_lossless_equivalent",
    "load_jsonl",
    "lossless_violations",
]
