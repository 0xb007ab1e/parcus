"""Quality judges for evaluating lossy/generative transforms (memory injection, compaction).

Lossy tiers can't use the lossless/filler invariants — they change content by design — so their
"equal quality" must be *judged*. Two judges are provided behind one :class:`QualityJudge`
interface:

* :class:`KeywordRecallJudge` — deterministic, model-free, CI-safe: does the candidate preserve
  the required information (substring recall of must-include phrases)?
* :class:`LLMJudge` — an offline adapter that delegates the verdict to an injected completion
  function (so it is testable with a fake and never hard-wires a network call into the suite).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["KeywordRecallJudge", "LLMJudge", "QualityJudge", "QualityVerdict"]


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    """The outcome of a quality judgement.

    Args:
        passed: Whether the candidate met the quality bar.
        score: A score in ``[0.0, 1.0]`` (e.g. recall fraction).
        reason: Short human-readable explanation.
    """

    passed: bool
    score: float
    reason: str


@runtime_checkable
class QualityJudge(Protocol):
    """Judges whether a candidate text preserves required information."""

    def judge(self, candidate: str, must_include: tuple[str, ...]) -> QualityVerdict:
        """Return a verdict on whether ``candidate`` preserves ``must_include``."""
        ...


class KeywordRecallJudge:
    """Deterministic recall judge: fraction of ``must_include`` phrases present in the candidate.

    Args:
        threshold: Minimum recall (in ``[0.0, 1.0]``) required to pass (default 1.0 = all).
    """

    def __init__(self, threshold: float = 1.0) -> None:
        """Initialise with the pass threshold."""
        self._threshold = threshold

    def judge(self, candidate: str, must_include: tuple[str, ...]) -> QualityVerdict:
        """Score recall of the required phrases (case-insensitive substring match)."""
        if not must_include:
            return QualityVerdict(passed=True, score=1.0, reason="no requirements")
        haystack = candidate.lower()
        present = [phrase for phrase in must_include if phrase.lower() in haystack]
        score = len(present) / len(must_include)
        passed = score >= self._threshold
        missing = [phrase for phrase in must_include if phrase.lower() not in haystack]
        reason = "all required phrases present" if passed else f"missing: {missing}"
        return QualityVerdict(passed=passed, score=score, reason=reason)


class LLMJudge:
    """Delegates the verdict to an injected completion function (offline-testable adapter).

    The ``complete`` callable receives a prompt and returns the model's reply; a reply starting
    with ``yes`` is a pass. Wiring ``complete`` to a real provider is the caller's concern and is
    kept out of the test suite (no network) — this judge is for occasional offline evaluation.

    Args:
        complete: A function mapping a prompt to a model reply.
    """

    def __init__(self, complete: Callable[[str], str]) -> None:
        """Initialise with the completion function."""
        self._complete = complete

    def judge(self, candidate: str, must_include: tuple[str, ...]) -> QualityVerdict:
        """Ask the model whether ``candidate`` preserves the required information."""
        prompt = (
            "Answer strictly 'yes' or 'no'. Does the TEXT contain all of these facts: "
            f"{list(must_include)}?\n\nTEXT:\n{candidate}"
        )
        reply = self._complete(prompt).strip().lower()
        passed = reply.startswith("yes")
        return QualityVerdict(passed=passed, score=1.0 if passed else 0.0, reason=reply[:80])
