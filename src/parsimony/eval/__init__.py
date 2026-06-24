"""Token-savings measurement harness with a lossless no-regression gate.

The headline metric — **% input-token reduction at equal quality** — is produced here.
For the always-on lossless tier, "equal quality" is enforced model-free by
:mod:`parsimony.eval.equivalence`; lossy tiers will plug a quality judge into
:func:`parsimony.eval.runner.evaluate` (``check_equivalence=False``) in a later milestone.
"""

from parsimony.eval.dataset import Sample, load_jsonl
from parsimony.eval.equivalence import is_lossless_equivalent, lossless_violations
from parsimony.eval.metrics import EvalReport, SampleResult
from parsimony.eval.runner import evaluate
from parsimony.eval.samples import BUILTIN_SAMPLES

__all__ = [
    "BUILTIN_SAMPLES",
    "EvalReport",
    "Sample",
    "SampleResult",
    "evaluate",
    "is_lossless_equivalent",
    "load_jsonl",
    "lossless_violations",
]
