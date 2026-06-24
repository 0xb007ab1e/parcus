"""Model-free term extraction for lexical retrieval over the memory graph."""

from __future__ import annotations

import re

__all__ = ["STOPWORDS", "extract_terms"]

# Small, generic stop-list; the goal is to drop low-signal tokens, not to be exhaustive.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "who",
        "did",
        "yes",
        "this",
        "that",
        "with",
        "from",
        "they",
        "have",
        "your",
        "will",
        "what",
        "when",
        "into",
        "than",
        "then",
        "them",
        "some",
        "such",
        "only",
        "also",
        "been",
        "were",
        "would",
        "could",
        "should",
        "about",
        "there",
        "their",
        "which",
        "while",
        "these",
        "those",
    }
)

_TOKEN = re.compile(r"[a-z0-9_]+")


def extract_terms(text: str, *, min_length: int = 3) -> frozenset[str]:
    """Return the set of normalised content terms in ``text``.

    Lower-cases, splits on non-alphanumeric boundaries, drops stop-words and tokens shorter
    than ``min_length``. Deterministic and dependency-free.

    Args:
        text: The text to extract terms from.
        min_length: Minimum token length to keep.

    Returns:
        A set of distinct content terms.
    """
    return frozenset(
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) >= min_length and token not in STOPWORDS
    )
