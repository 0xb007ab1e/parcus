"""Classify message text into mutable (prose) and immutable (code) spans.

Compression may only alter mutable spans, so we must mark anything meaning-bearing as
immutable first. For M1 this protects fenced code blocks (```` ``` ````); inline code, indented
code, and other immutable classes (paths/URLs/quotes) are handled within the lossless pass's
conservative whitespace rules and will be expanded in a later milestone.
"""

from __future__ import annotations

import re

from parsimony.model import Span

__all__ = ["classify_spans"]

# Matched fenced pairs only. An unterminated fence is left as prose (the lossless pass only
# trims trailing whitespace / blank lines, so the risk to such text is minimal); proper
# handling of unterminated fences is a later-milestone refinement.
_FENCE = re.compile(r"```.*?```", re.DOTALL)


def classify_spans(text: str) -> tuple[Span, ...]:
    """Split ``text`` into alternating mutable prose and immutable fenced-code spans.

    Args:
        text: The raw message text.

    Returns:
        A non-empty tuple of spans whose concatenation reproduces ``text`` exactly.
    """
    spans: list[Span] = []
    pos = 0
    for match in _FENCE.finditer(text):
        if match.start() > pos:
            spans.append(Span(text[pos : match.start()], mutable=True))
        spans.append(Span(match.group(0), mutable=False))
        pos = match.end()
    if pos < len(text):
        spans.append(Span(text[pos:], mutable=True))
    if not spans:
        spans.append(Span(text, mutable=True))
    return tuple(spans)
