"""Back-compatible re-export of the compression invariants.

The model-free correctness invariants now live in :mod:`parsimony.invariants` so the
compressors can self-verify at runtime without an import cycle. This module re-exports them
unchanged for existing callers and the eval gate.
"""

from parsimony.invariants import (
    filler_violations,
    is_filler_equivalent,
    is_lossless_equivalent,
    lossless_violations,
    removed_only_allowed,
    word_sequence_equal,
)

__all__ = [
    "filler_violations",
    "is_filler_equivalent",
    "is_lossless_equivalent",
    "lossless_violations",
    "removed_only_allowed",
    "word_sequence_equal",
]
