"""Token counting with an exact tokenizer when available and a stable heuristic fallback.

Token *measurement* drives every savings claim, so it must be deterministic. When the exact
provider tokenizer (e.g. ``tiktoken``) is installed it is used; otherwise a stable
characters-per-token heuristic is used. The heuristic is intentionally simple and never makes
a network call (a tokenizer that phoned home would defeat the project's purpose).
"""

from __future__ import annotations

__all__ = ["HeuristicTokenizer", "default_tokenizer"]

# Empirically ~4 characters per token for English + code across common BPE vocabularies.
_CHARS_PER_TOKEN = 4


class HeuristicTokenizer:
    """A deterministic, dependency-free token counter (~4 chars/token).

    Implements :class:`parsimony.ports.TokenizerPort`. Used as a fallback and in tests so the
    core never requires a model download.
    """

    def count(self, text: str, model: str | None = None) -> int:
        """Return an estimated token count for ``text``.

        The ``model`` argument is accepted for interface compatibility but ignored by the
        heuristic (which is model-independent).
        """
        if not text:
            return 0
        # Round up so non-empty text always counts as >= 1 token.
        return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def default_tokenizer() -> HeuristicTokenizer:
    """Return the default tokenizer.

    Currently the heuristic; a later change will lazily prefer an installed exact tokenizer
    (``tiktoken`` for OpenAI, Anthropic's token counter) behind this same factory so callers
    do not change.
    """
    return HeuristicTokenizer()
