"""Model-free correctness invariants for the compression stages.

Pure functions that depend only on the canonical model. They serve two roles:

* the **eval gate** (offline, in :mod:`parcus.eval`), and
* each compressor's **runtime self-check** — a live correctness monitor that records, per
  request, whether the transform stayed within its guarantee (lossless = whitespace-only;
  filler = only allow-listed tokens removed). Healthy = ~100%; a dip flags a bug.

They live here (not in :mod:`parcus.eval`) so the compressors can import them without an
import cycle.
"""

from __future__ import annotations

import string

from parcus.model import CanonicalRequest

__all__ = [
    "filler_violations",
    "is_filler_equivalent",
    "is_lossless_equivalent",
    "lossless_violations",
    "removed_only_allowed",
    "word_sequence_equal",
]

_PUNCTUATION = string.punctuation


def word_sequence_equal(a: str, b: str) -> bool:
    """Return whether two strings are equal ignoring all whitespace differences."""
    return a.split() == b.split()


def removed_only_allowed(original: str, compressed: str, allowed: frozenset[str]) -> bool:
    """Return whether ``compressed`` is ``original`` with only allow-listed tokens deleted.

    Kept tokens must appear unchanged and in their original order; every original token absent
    from the output must normalise (lower-case, strip punctuation) to a member of ``allowed``.
    Nothing may be added, altered, or reordered.

    Args:
        original: The pre-compression text.
        compressed: The post-compression text.
        allowed: The set of normalised tokens permitted to be removed.

    Returns:
        True if the transformation removed only allowed tokens, else False.
    """
    out_tokens = compressed.split()
    index = 0
    for token in original.split():
        if index < len(out_tokens) and token == out_tokens[index]:
            index += 1
        elif token.strip(_PUNCTUATION).lower() in allowed:
            continue  # an allowed filler was removed
        else:
            return False  # a kept token was altered/reordered or a non-filler was dropped
    return index == len(out_tokens)  # every output token was matched (nothing added)


def lossless_violations(original: CanonicalRequest, compressed: CanonicalRequest) -> list[str]:
    """Return a list of human-readable lossless-invariant violations (empty when equivalent).

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


def filler_violations(
    original: CanonicalRequest,
    compressed: CanonicalRequest,
    allowed: frozenset[str],
) -> list[str]:
    """Return violations of the Tier-1 invariant: only ``allowed`` filler tokens were removed.

    Like :func:`lossless_violations` but mutable spans (and ``system``) may differ by the
    deletion of allow-listed tokens (verified with :func:`removed_only_allowed`); immutable
    spans must still be byte-identical and structure unchanged.

    Args:
        original: The request before compression.
        compressed: The request after the filler pass.
        allowed: The normalised tokens permitted to be removed.

    Returns:
        Zero or more violation descriptions; a non-empty list means a regression.
    """
    issues: list[str] = []

    if (original.system is None) != (compressed.system is None):
        issues.append("system presence changed")
    elif (
        original.system is not None
        and compressed.system is not None
        and not removed_only_allowed(original.system, compressed.system, allowed)
    ):
        issues.append("system content changed beyond allowed fillers")

    if len(original.messages) != len(compressed.messages):
        issues.append("message count changed")
        return issues

    for index, (orig_msg, comp_msg) in enumerate(
        zip(original.messages, compressed.messages, strict=True)
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
            elif not removed_only_allowed(orig_span.text, comp_span.text, allowed):
                issues.append(f"message {index}: mutable span changed beyond allowed fillers")

    return issues


def is_filler_equivalent(
    original: CanonicalRequest,
    compressed: CanonicalRequest,
    allowed: frozenset[str],
) -> bool:
    """Return whether ``compressed`` is ``original`` with only ``allowed`` fillers removed."""
    return not filler_violations(original, compressed, allowed)
