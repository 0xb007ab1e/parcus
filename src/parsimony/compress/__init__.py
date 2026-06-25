"""Tiered, guardrailed request compression.

Tiers (see ``PLAN.md`` §4):

* **Tier 0 — lossless** (:class:`~parsimony.compress.lossless.LosslessCompressor`): default on,
  zero semantic risk. Implemented here.
* **Tier 1 — filler removal** (:class:`~parsimony.compress.filler.FillerCompressor`): opt-in,
  guardrailed, lossy — removes only allow-listed discourse tokens (model-free invariant).
* **Tier 2 — learned** (:class:`~parsimony.compress.learned.LearnedCompressor`): opt-in, **local**
  model; lossy/semantic with no runtime invariant — gated offline by an answer-preservation judge.

Every compressor only ever alters spans marked ``mutable`` and must fail open: on any internal
error it returns the request unchanged.
"""

from parsimony.compress.chain import ChainCompressor
from parsimony.compress.filler import DEFAULT_FILLERS, FillerCompressor
from parsimony.compress.learned import LearnedCompressor, LLMLinguaReducer, TokenReducer
from parsimony.compress.lossless import LosslessCompressor
from parsimony.compress.null import NullCompressor

__all__ = [
    "DEFAULT_FILLERS",
    "ChainCompressor",
    "FillerCompressor",
    "LLMLinguaReducer",
    "LearnedCompressor",
    "LosslessCompressor",
    "NullCompressor",
    "TokenReducer",
]
