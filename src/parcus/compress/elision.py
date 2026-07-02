"""Lossy elision of stale tool-result payloads from structured history (M1d slice 3).

Old ``tool_result`` blocks (a file dump or command output from many turns ago) are re-sent verbatim
every turn and dominate history size, yet it's the *recent* turns the model actually needs. This
pass replaces the payload of ``tool_result`` blocks in messages older than a keep-recent window
with a compact stub, **preserving each block's ``tool_use_id``/``is_error``** so tool pairing stays
intact.

Lossy → **opt-in and off by default**, validated on the answer-preservation eval like the other
lossy tiers (``parcus eval --judged``). It only affects structured (``raw``) messages the parser
carries under ``parse_structured``; plain-text turns and recent turns are untouched. Implements
:class:`parcus.ports.CompressorPort` and **fails open** (returns the request unchanged on error).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from parcus.model import CanonicalRequest, CompressionStats, Message
from parcus.ports import TokenizerPort
from parcus.tokenize import default_tokenizer

__all__ = ["ToolResultElider"]

_STEP_NAME = "elision"
_DEFAULT_STUB = "[tool result elided by parcus]"


class ToolResultElider:
    """Replace stale ``tool_result`` payloads with a stub. A lossy, opt-in compression pass."""

    def __init__(
        self,
        *,
        keep_recent: int = 4,
        stub: str = _DEFAULT_STUB,
        tokenizer: TokenizerPort | None = None,
    ) -> None:
        """Configure the elider.

        Args:
            keep_recent: Number of most-recent messages left untouched (their tool results are
                usually still load-bearing). Messages before this window are eligible for elision.
            stub: The compact placeholder that replaces an elided tool-result payload.
            tokenizer: Token counter for the stats (defaults to the shared tokenizer).
        """
        self._keep_recent = max(0, keep_recent)
        self._stub = stub
        self._tokenizer = tokenizer or default_tokenizer()

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Elide stale tool-result payloads and return the request plus this pass's stats.

        Fails open: on any internal error the original ``request`` is returned with empty stats.
        """
        try:
            before = self._tokenizer.count(request.text, request.model)
            cutoff = len(request.messages) - self._keep_recent
            new_messages: list[Message] = []
            touched = 0
            for index, message in enumerate(request.messages):
                if message.raw is not None and index < cutoff:
                    new_raw, count = _elide_tool_results(message.raw, self._stub)
                    touched += count
                    if new_raw is message.raw:
                        new_messages.append(message)
                    else:
                        new_messages.append(Message(role=message.role, spans=(), raw=new_raw))
                else:
                    new_messages.append(message)
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


def _elide_tool_results(raw: dict[str, Any], stub: str) -> tuple[dict[str, Any], int]:
    """Return a copy of a structured message with tool-result payloads stubbed, and the count.

    Returns the same object (and 0) when the content isn't a block list or nothing was elided.
    Other block types, and each block's ``tool_use_id``/``is_error``, are preserved verbatim.
    """
    content = raw.get("content")
    if not isinstance(content, list):
        return raw, 0
    touched = 0
    new_content: list[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            payload = block.get("content")
            # Only elide a payload that's actually larger than the stub — replacing a tiny result
            # with the stub would *cost* tokens (never-cost-more); also skips empty/already-elided.
            if payload is not None and payload != stub and _payload_len(payload) > len(stub):
                touched += 1
                block = {**block, "content": stub}
        new_content.append(block)
    if touched == 0:
        return raw, 0
    return {**raw, "content": new_content}, touched


def _payload_len(payload: object) -> int:
    """A cheap character-length proxy for a tool-result payload (string or block list)."""
    return (
        len(payload) if isinstance(payload, str) else len(json.dumps(payload, ensure_ascii=False))
    )
