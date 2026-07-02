"""Tiered, guardrailed request compression.

Tiers (see ``PLAN.md`` §4):

* **Tier 0 — lossless** (:class:`~parcus.compress.lossless.LosslessCompressor`): default on,
  zero semantic risk. Implemented here.
* **Tier 1 — filler removal** (:class:`~parcus.compress.filler.FillerCompressor`): opt-in,
  guardrailed, lossy — removes only allow-listed discourse tokens (model-free invariant).
* **Tier 2 — learned** (:class:`~parcus.compress.learned.LearnedCompressor`): opt-in, **local**
  model; lossy/semantic with no runtime invariant — gated offline by an answer-preservation judge.

Every compressor only ever alters spans marked ``mutable`` and must fail open: on any internal
error it returns the request unchanged.
"""

from parcus.compress.chain import ChainCompressor
from parcus.compress.elision import ToolResultElider
from parcus.compress.filler import AGGRESSIVE_FILLERS, DEFAULT_FILLERS, FillerCompressor
from parcus.compress.learned import LearnedCompressor, LLMLinguaReducer, TokenReducer
from parcus.compress.lossless import LosslessCompressor
from parcus.compress.null import NullCompressor

__all__ = [
    "AGGRESSIVE_FILLERS",
    "DEFAULT_FILLERS",
    "ChainCompressor",
    "FillerCompressor",
    "LLMLinguaReducer",
    "LearnedCompressor",
    "LosslessCompressor",
    "NullCompressor",
    "TokenReducer",
    "ToolResultElider",
]
