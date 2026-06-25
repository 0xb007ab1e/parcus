"""Tier-1 filler removal — drop allow-listed discourse/filler tokens from prose spans.

This pass is **lossy** and **opt-in (off by default)**. It removes only whole tokens whose
normalised form is in a curated allow-list, and only within **mutable** prose spans (fenced
code, quotes, tool JSON, and other immutable spans are never touched). It never reorders or
alters kept tokens.

The guardrail (:mod:`parcus.eval.equivalence`) proves model-free that *only allow-listed
fillers were removed* — i.e. no real content was dropped, added, or reordered. The **semantic**
safety of the allow-list itself is a human-curated concern; the default set is intentionally
conservative, and a quality judge for aggressive sets is planned (see ``PLAN.md`` Tier-2).
"""

from __future__ import annotations

import string

from parcus.compress.sampling import VerifySampler
from parcus.invariants import is_filler_equivalent
from parcus.model import CanonicalRequest, CompressionStats, Message, Span
from parcus.ports import TokenizerPort
from parcus.spans import classify_spans
from parcus.tokenize import default_tokenizer

__all__ = ["AGGRESSIVE_FILLERS", "DEFAULT_FILLERS", "FillerCompressor", "strip_fillers"]

_STEP_NAME = "filler"
_PUNCTUATION = string.punctuation

# Conservative, opt-in default: politeness markers + common intensifiers/discourse hedges that
# are safe to drop in imperative/instructional text. Tunable; kept small on purpose.
DEFAULT_FILLERS: frozenset[str] = frozenset(
    {
        "please",
        "kindly",
        "really",
        "very",
        "actually",
        "basically",
        "simply",
        "essentially",
        "literally",
        "honestly",
        "just",
        "quite",
    }
)

# Larger, opt-in set (superset of DEFAULT) adding more discourse hedges/intensifiers/fillers.
# Removal stays **structurally** safe — the model-free guardrail (``is_filler_equivalent``) still
# proves only allow-listed whole tokens were dropped, for any set. The added words carry more
# **semantic** risk than the conservative default, so validate this set with the offline quality
# judge (``parcus eval --filler --aggressive``; LLMJudge) before enabling it in production.
AGGRESSIVE_FILLERS: frozenset[str] = DEFAULT_FILLERS | frozenset(
    {
        "obviously",
        "clearly",
        "certainly",
        "definitely",
        "absolutely",
        "totally",
        "completely",
        "particularly",
        "specifically",
        "generally",
        "typically",
        "perhaps",
        "maybe",
        "somewhat",
        "rather",
        "fairly",
        "pretty",
        "truly",
        "indeed",
        "surely",
        "arguably",
        "presumably",
    }
)


def _normalise(token: str) -> str:
    """Lower-case and strip surrounding punctuation for filler-membership testing."""
    return token.strip(_PUNCTUATION).lower()


def _filter_prose(text: str, fillers: frozenset[str]) -> str:
    """Drop filler tokens from a pure-prose string, preserving line breaks and kept tokens."""
    return "\n".join(
        " ".join(tok for tok in line.split() if _normalise(tok) not in fillers)
        for line in text.split("\n")
    )


def strip_fillers(text: str, fillers: frozenset[str] = DEFAULT_FILLERS) -> str:
    """Remove fillers from the prose (mutable) spans of ``text``, preserving fenced code."""
    return "".join(
        span.text if not span.mutable else _filter_prose(span.text, fillers)
        for span in classify_spans(text)
    )


class FillerCompressor:
    """Remove allow-listed filler tokens from mutable spans. Implements ``CompressorPort``.

    Args:
        fillers: The allow-list of removable tokens (default :data:`DEFAULT_FILLERS`).
        tokenizer: Token counter for measurement (default heuristic).
        verify_sample: Fraction of calls on which to run the invariant self-check (default 1.0).
    """

    def __init__(
        self,
        fillers: frozenset[str] | None = None,
        tokenizer: TokenizerPort | None = None,
        *,
        verify_sample: float = 1.0,
    ) -> None:
        """Initialise with an optional custom filler set, tokenizer, and self-check sampler."""
        self._fillers = fillers if fillers is not None else DEFAULT_FILLERS
        self._tokenizer = tokenizer or default_tokenizer()
        self._sampler = VerifySampler(verify_sample)

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Return the request with prose fillers removed, plus stats (fails open)."""
        try:
            before = self._tokenizer.count(request.text, request.model)
            touched = 0
            new_messages: list[Message] = []
            for message in request.messages:
                new_spans: list[Span] = []
                for span in message.spans:
                    if span.mutable:
                        filtered = _filter_prose(span.text, self._fillers)
                        if filtered != span.text:
                            touched += 1
                            new_spans.append(span.with_text(filtered))
                            continue
                    new_spans.append(span)
                new_messages.append(Message(role=message.role, spans=tuple(new_spans)))

            new_system = request.system
            if request.system is not None:
                rebuilt = strip_fillers(request.system, self._fillers)
                if rebuilt != request.system:
                    touched += 1
                    new_system = rebuilt

            new_request = CanonicalRequest(
                dialect=request.dialect,
                model=request.model,
                messages=tuple(new_messages),
                system=new_system,
                stream=request.stream,
                tools_json=request.tools_json,
            )
            after = self._tokenizer.count(new_request.text, new_request.model)
            stats = CompressionStats(
                step=_STEP_NAME,
                tokens_before=before,
                tokens_after=after,
                spans_touched=touched,
                ok=(
                    is_filler_equivalent(request, new_request, self._fillers)  # sampled
                    if self._sampler.should_verify()
                    else None
                ),
            )
            return new_request, (stats,)
        except Exception:
            # Fail open: never break the request path to save tokens.
            return request, ()
