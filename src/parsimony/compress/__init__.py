"""Tiered, guardrailed request compression.

Tiers (see ``PLAN.md`` §4):

* **Tier 0 — lossless** (:class:`~parsimony.compress.lossless.LosslessCompressor`): default on,
  zero semantic risk. Implemented here.
* **Tier 1 — filler removal**: opt-in, guardrailed, lossy (future milestone M2).
* **Tier 2 — learned**: opt-in, local model (future milestone M4).

Every compressor only ever alters spans marked ``mutable`` and must fail open: on any internal
error it returns the request unchanged.
"""

from parsimony.compress.lossless import LosslessCompressor
from parsimony.compress.null import NullCompressor

__all__ = ["LosslessCompressor", "NullCompressor"]
