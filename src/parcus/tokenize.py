"""Token counting with an exact BPE tokenizer when available and a stable heuristic fallback.

Token *measurement* drives every savings claim, so it must be deterministic and as accurate as
possible. When ``tiktoken`` (a core dependency) can load an encoding, it is used — exact for
OpenAI-family models, and a close approximation for other BPE tokenizers (Groq/Llama, Anthropic).
Otherwise a stable characters-per-token heuristic is used. Neither ever makes a network call at
request time beyond tiktoken's one-time vocab load (cached); a tokenizer that phoned home per
request would defeat the project's purpose.

**Why the switch matters (validated against a real provider):** the old 4-chars/token heuristic
over-counted exactly the whitespace/filler parcus removes, so it *overstated* savings ~1.5-2.2x
versus the provider's billed tokens. A real BPE encoding tracks the provider's tokenizer with a
near-constant offset (the provider's fixed chat-template overhead -- role/BOS markers -- which the
message text does not include), so the **saved-token delta is accurate** even though the absolute
count runs a little under the provider's billed prompt. See ``docs/validation``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken

__all__ = ["HeuristicTokenizer", "TiktokenTokenizer", "default_tokenizer"]

# Empirically ~4 characters per token for English + code across common BPE vocabularies.
_CHARS_PER_TOKEN = 4
# Stable default encoding for models tiktoken doesn't map (Llama/Groq, Anthropic, …). Widely
# used and vocab-stable; a close approximation whose per-request offset is near-constant, so
# compression deltas remain accurate.
_DEFAULT_ENCODING = "cl100k_base"


class HeuristicTokenizer:
    """A deterministic, dependency-free token counter (~4 chars/token).

    Implements :class:`parcus.ports.TokenizerPort`. Used as a fallback and in tests so the
    core never requires a model/vocab download.
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


@lru_cache(maxsize=64)
def _encoding(model: str | None) -> tiktoken.Encoding | None:
    """Return a tiktoken encoding for ``model`` (or a stable default), or ``None`` if unavailable.

    Cached per model. Never raises: a missing ``tiktoken``, an unmappable model, or a vocab that
    cannot be loaded (e.g. offline) all yield ``None`` so the caller falls back to the heuristic.
    """
    try:
        import tiktoken
    except Exception:  # tiktoken not importable
        return None
    try:
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                pass  # unknown model → fall through to the default encoding
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:  # pragma: no cover - vocab unavailable/offline; defensive
        return None


class TiktokenTokenizer:
    """Exact-BPE token counter via ``tiktoken`` with a heuristic fallback.

    Implements :class:`parcus.ports.TokenizerPort`. Counts the *message text*; providers add a
    small fixed chat-template overhead not observable here, so absolute counts run slightly under
    the billed prompt while the saved-token delta stays accurate. Falls back to
    :class:`HeuristicTokenizer` when tiktoken or the vocab is unavailable, so counting never
    fails or blocks (fail open).
    """

    def __init__(self) -> None:
        """Initialise with a heuristic fallback for when no encoding is available."""
        self._fallback = HeuristicTokenizer()

    def count(self, text: str, model: str | None = None) -> int:
        """Return the exact BPE token count for ``text`` (heuristic if no encoding loads)."""
        if not text:
            return 0
        enc = _encoding(model)
        if enc is None:
            return self._fallback.count(text, model)
        try:
            # disallowed_special=() treats special-token-looking text as ordinary (never raises).
            return len(enc.encode(text, disallowed_special=()))
        except Exception:  # pragma: no cover - defensive; encode is robust with disallowed_special
            return self._fallback.count(text, model)


def default_tokenizer() -> TiktokenTokenizer:
    """Return the default tokenizer: exact tiktoken BPE when available, else the heuristic."""
    return TiktokenTokenizer()
