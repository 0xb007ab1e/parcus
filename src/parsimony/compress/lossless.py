"""Tier-0 lossless compression — safe whitespace normalisation of mutable spans only.

This pass carries **zero semantic risk**: it never touches immutable spans (code, paths,
URLs, quoted text, tool JSON, the trailing instruction) and, within mutable spans, only
removes whitespace that has no effect on meaning:

* strip trailing whitespace from each line;
* collapse three-or-more consecutive newlines to a single blank line;
* trim leading/trailing blank lines of the span.

More aggressive (but still lossless) normalisations and verbatim-block de-duplication are
planned; each will be added only behind the eval harness. The compressor **fails open**: any
unexpected error yields the original request unchanged.
"""

from __future__ import annotations

import re

from parsimony.model import CanonicalRequest, CompressionStats, Message, Span
from parsimony.ports import TokenizerPort
from parsimony.spans import classify_spans
from parsimony.tokenize import default_tokenizer

__all__ = ["LosslessCompressor", "normalise_code_aware", "normalise_whitespace"]

_STEP_NAME = "lossless"
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalise_whitespace(text: str) -> str:
    """Return ``text`` with meaning-preserving whitespace removed.

    Strips trailing whitespace from every line (including the final line) and collapses runs
    of 3+ newlines to a single blank line. Interior single spaces and leading indentation are
    preserved. Boundary blank-line runs are *collapsed*, not fully removed — applied per span,
    fully trimming them would merge prose into an adjacent code block and lose its separation.

    Args:
        text: The span text to normalise.

    Returns:
        The normalised text (possibly identical to the input).
    """
    out = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", out)


def normalise_code_aware(text: str) -> str:
    """Normalise only the prose (mutable) spans of ``text``, preserving fenced code verbatim."""
    return "".join(
        normalise_whitespace(span.text) if span.mutable else span.text
        for span in classify_spans(text)
    )


class LosslessCompressor:
    """Apply :func:`normalise_whitespace` to mutable spans only.

    Implements :class:`parsimony.ports.CompressorPort`.

    Args:
        tokenizer: Token counter used to measure savings. Defaults to the heuristic tokenizer.
    """

    def __init__(self, tokenizer: TokenizerPort | None = None) -> None:
        """Initialise the compressor with an optional token counter."""
        self._tokenizer = tokenizer or default_tokenizer()

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Return a whitespace-normalised request plus this pass's stats.

        Fails open: on any internal error the original ``request`` is returned with empty
        stats (never raises).
        """
        try:
            before = self._tokenizer.count(request.text, request.model)
            touched = 0
            new_messages: list[Message] = []
            for message in request.messages:
                new_spans: list[Span] = []
                for span in message.spans:
                    if span.mutable:
                        normalised = normalise_whitespace(span.text)
                        if normalised != span.text:
                            touched += 1
                            new_spans.append(span.with_text(normalised))
                            continue
                    new_spans.append(span)
                new_messages.append(Message(role=message.role, spans=tuple(new_spans)))

            new_system = request.system
            if request.system is not None:
                rebuilt = normalise_code_aware(request.system)
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
            )
            return new_request, (stats,)
        except Exception:
            # Fail open: never break the request path to save tokens.
            return request, ()
