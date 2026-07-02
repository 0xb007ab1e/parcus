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
from typing import Any

from parcus.compress.sampling import VerifySampler
from parcus.invariants import is_lossless_equivalent
from parcus.model import CanonicalRequest, CompressionStats, Message, Span
from parcus.ports import TokenizerPort
from parcus.spans import classify_spans
from parcus.tokenize import default_tokenizer

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


def _normalise_text_blocks(raw: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Whitespace-normalise the ``text`` blocks of a structured message (M1d slice 2).

    Only ``{"type": "text", "text": <str>, ...}`` blocks are touched (code-fence-aware, lossless);
    every other block (tool_use/tool_result/image) is reproduced verbatim. Returns the message dict
    — the same object when its content isn't a block list or nothing changed — and the number of
    text blocks altered.
    """
    content = raw.get("content")
    if not isinstance(content, list):
        return raw, 0
    touched = 0
    new_content: list[Any] = []
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            normalised = normalise_code_aware(block["text"])
            if normalised != block["text"]:
                touched += 1
                block = {**block, "text": normalised}
        new_content.append(block)
    if touched == 0:
        return raw, 0
    return {**raw, "content": new_content}, touched


class LosslessCompressor:
    """Apply :func:`normalise_whitespace` to mutable spans only.

    Implements :class:`parcus.ports.CompressorPort`.

    Args:
        tokenizer: Token counter used to measure savings. Defaults to the heuristic tokenizer.
        verify_sample: Fraction of calls on which to run the invariant self-check (default 1.0
            = always). Lower it to trim per-request overhead at high throughput.
    """

    def __init__(
        self, tokenizer: TokenizerPort | None = None, *, verify_sample: float = 1.0
    ) -> None:
        """Initialise the compressor with an optional token counter and self-check sampler."""
        self._tokenizer = tokenizer or default_tokenizer()
        self._sampler = VerifySampler(verify_sample)

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
                if message.raw is not None:
                    # Structured content: normalise text blocks (slice 2); other blocks verbatim.
                    new_raw, block_touched = _normalise_text_blocks(message.raw)
                    touched += block_touched
                    if new_raw is message.raw:
                        new_messages.append(message)
                    else:
                        new_messages.append(Message(role=message.role, spans=(), raw=new_raw))
                    continue
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
                ok=(
                    is_lossless_equivalent(request, new_request)  # live self-check (sampled)
                    if self._sampler.should_verify()
                    else None
                ),
            )
            return new_request, (stats,)
        except Exception:
            # Fail open: never break the request path to save tokens.
            return request, ()
