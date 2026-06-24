"""The lossless no-regression invariant: compression may only remove whitespace.

For Tier-0 (lossless) compression, "equal quality" is a checkable property — no model needed:

* every **immutable** span (code, etc.) is byte-for-byte identical, and
* every **mutable** span is identical once whitespace is ignored (same word sequence), and
* roles, message count, and ``system`` presence are unchanged.

Any violation is a correctness regression and fails the eval gate. Lossy tiers (filler/learned)
break this invariant by design and will instead be scored by a quality judge plugged into the
eval runner; this checker remains the gate for the always-on lossless pass.
"""

from __future__ import annotations

from parsimony.model import CanonicalRequest

__all__ = ["is_lossless_equivalent", "lossless_violations", "word_sequence_equal"]


def word_sequence_equal(a: str, b: str) -> bool:
    """Return whether two strings are equal ignoring all whitespace differences."""
    return a.split() == b.split()


def lossless_violations(original: CanonicalRequest, compressed: CanonicalRequest) -> list[str]:
    """Return a list of human-readable invariant violations (empty when equivalent).

    Args:
        original: The request before compression.
        compressed: The request after a lossless compression pass.

    Returns:
        Zero or more violation descriptions; a non-empty list means a regression.
    """
    issues: list[str] = []

    if (original.system is None) != (compressed.system is None):
        issues.append("system presence changed")
    elif (
        original.system is not None
        and compressed.system is not None
        and not word_sequence_equal(original.system, compressed.system)
    ):
        issues.append("system content changed beyond whitespace")

    if len(original.messages) != len(compressed.messages):
        issues.append("message count changed")
        return issues

    for index, (orig_msg, comp_msg) in enumerate(
        zip(original.messages, compressed.messages, strict=True)  # lengths checked above
    ):
        if orig_msg.role != comp_msg.role:
            issues.append(f"message {index}: role changed")
        if len(orig_msg.spans) != len(comp_msg.spans):
            issues.append(f"message {index}: span structure changed")
            continue
        for orig_span, comp_span in zip(orig_msg.spans, comp_msg.spans, strict=True):
            if not orig_span.mutable:
                if orig_span.text != comp_span.text:
                    issues.append(f"message {index}: immutable span altered")
            elif not word_sequence_equal(orig_span.text, comp_span.text):
                issues.append(f"message {index}: mutable span content changed beyond whitespace")

    return issues


def is_lossless_equivalent(original: CanonicalRequest, compressed: CanonicalRequest) -> bool:
    """Return whether ``compressed`` preserves ``original``'s meaning (lossless invariant)."""
    return not lossless_violations(original, compressed)
