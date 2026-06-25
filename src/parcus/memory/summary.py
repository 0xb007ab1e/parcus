"""Summarisers for Track C conversation compaction.

Track C replaces verbose older history with one compact summary block (query-independent,
unlike Track B's per-query retrieval). Two adapters behind one :class:`Summarizer`:

* :class:`ExtractiveSummarizer` — model-free, deterministic, CI-safe: keep the lead of each
  distinct older turn, in order, up to a cap. No model, no network (the default).
* :class:`LLMSummarizer` — an offline adapter that delegates abstractive summarisation to an
  injected completion function (testable with a fake; no network wired into the suite). Using a
  remote model to summarise would be an outbound call, so it is opt-in and offline-only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

__all__ = ["ExtractiveSummarizer", "LLMSummarizer", "Summarizer"]

_SENTENCE_ENDS = (". ", "? ", "! ")
_MAX_LEAD = 200


@runtime_checkable
class Summarizer(Protocol):
    """Condenses a sequence of texts into a compact summary string."""

    def summarize(self, texts: Sequence[str], *, max_items: int = 5) -> str:
        """Return a compact summary of ``texts`` (at most ``max_items`` points)."""
        ...


def _lead(text: str) -> str:
    """Return the first sentence of the first non-empty line, capped in length."""
    stripped = text.strip()
    if not stripped:
        return ""
    line = stripped.splitlines()[0].strip()
    for end in _SENTENCE_ENDS:
        index = line.find(end)
        if index != -1:
            line = line[: index + 1]
            break
    return line[:_MAX_LEAD].strip()


class ExtractiveSummarizer:
    """Model-free summariser: distinct turn-leads in order. Implements ``Summarizer``."""

    def summarize(self, texts: Sequence[str], *, max_items: int = 5) -> str:
        """Return up to ``max_items`` distinct, order-preserving turn leads as bullet points."""
        leads: list[str] = []
        seen: set[str] = set()
        for text in texts:
            lead = _lead(text)
            if not lead:
                continue
            key = lead.lower()
            if key in seen:
                continue
            seen.add(key)
            leads.append(lead)
            if len(leads) >= max_items:
                break
        return "\n".join(f"- {lead}" for lead in leads)


class LLMSummarizer:
    """Abstractive summariser over an injected completion function (offline-testable adapter).

    Args:
        complete: A function mapping a prompt to a model reply.
    """

    def __init__(self, complete: Callable[[str], str]) -> None:
        """Initialise with the completion function."""
        self._complete = complete

    def summarize(self, texts: Sequence[str], *, max_items: int = 5) -> str:
        """Ask the model to summarise ``texts`` into at most ``max_items`` points."""
        joined = "\n".join(f"- {text}" for text in texts)
        prompt = (
            f"Summarise the following conversation turns into at most {max_items} concise "
            f"bullet points, preserving concrete facts:\n\n{joined}"
        )
        return self._complete(prompt).strip()
