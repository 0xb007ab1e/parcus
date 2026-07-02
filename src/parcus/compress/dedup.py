"""Cross-turn verbatim dedup of repeated content within a request (Tier-2, lossy, opt-in).

Agentic payloads often re-paste the same large block across turns — a file dumped in one turn and
quoted again later, a repeated context block, boilerplate. A stateless provider can't reference
content from a *prior* API call, so dedup operates **within the one request**: content-addressed by
exact span text, it keeps the **first** occurrence of a substantial block and replaces later
byte-identical copies with a compact reference marker.

This is the one operation permitted to replace a repeated **immutable** block (a pasted file is a
fenced-code span): the content is preserved verbatim in its first occurrence, so only the redundant
copy is removed. Lossy → **opt-in, off by default**, validated on the answer-preservation eval like
the other lossy tiers. Never-cost-more (only blocks larger than the marker are eligible), fail-open,
and structured (``raw``) messages are left untouched in this slice.
"""

from __future__ import annotations

from dataclasses import replace

from parcus.model import CanonicalRequest, CompressionStats, Message, Span
from parcus.ports import TokenizerPort
from parcus.tokenize import default_tokenizer

__all__ = ["DedupCompressor"]

_STEP_NAME = "dedup"
_DEDUP_MARKER = "[identical block above — deduplicated by parcus]"


class DedupCompressor:
    """Replace a later byte-identical copy of a large block with a reference (opt-in, lossy)."""

    def __init__(
        self,
        *,
        min_chars: int = 200,
        marker: str = _DEDUP_MARKER,
        tokenizer: TokenizerPort | None = None,
    ) -> None:
        """Configure the deduper.

        Args:
            min_chars: Only blocks at least this many characters are eligible (avoids deduping
                noise). Clamped to be strictly larger than the marker so a replacement always
                shrinks the request (never-cost-more).
            marker: The reference text that replaces a repeated block.
            tokenizer: Token counter for the stats (defaults to the shared tokenizer).
        """
        self._marker = marker
        self._min_chars = max(min_chars, len(marker) + 1)
        self._tokenizer = tokenizer or default_tokenizer()

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Dedup repeated blocks across the request and return it plus this pass's stats.

        Fails open: on any internal error the original ``request`` is returned with empty stats.
        """
        try:
            before = self._tokenizer.count(request.text, request.model)
            seen: set[str] = set()
            touched = 0
            new_messages: list[Message] = []
            for message in request.messages:
                if message.raw is not None:
                    new_messages.append(message)  # structured content: dedup deferred
                    continue
                new_spans: list[Span] = []
                for span in message.spans:
                    if len(span.text) >= self._min_chars:
                        if span.text in seen:
                            touched += 1
                            new_spans.append(Span(self._marker, mutable=False))
                            continue
                        seen.add(span.text)
                    new_spans.append(span)
                new_messages.append(Message(role=message.role, spans=tuple(new_spans)))
            if touched == 0:
                return request, ()
            new_request = replace(request, messages=tuple(new_messages))
            after = self._tokenizer.count(new_request.text, new_request.model)
            return new_request, (
                CompressionStats(
                    step=_STEP_NAME,
                    tokens_before=before,
                    tokens_after=after,
                    spans_touched=touched,
                    ok=None,  # lossy: no model-free equivalence invariant
                ),
            )
        except Exception:
            return request, ()
