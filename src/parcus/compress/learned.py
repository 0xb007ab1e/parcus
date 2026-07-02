"""Tier-2 learned compression — drop low-information tokens with a **local** model (opt-in).

This pass is **lossy and semantic**: unlike Tier-0 (whitespace) and Tier-1 (allow-listed
fillers), it removes whatever a model judges low-information, so there is **no model-free
invariant** that proves equivalence. Its correctness is therefore validated **offline** by an
answer-preservation quality judge (``parcus.eval.quality.LLMJudge``) against the eval corpus
before an operator enables it — not by a runtime self-check. Accordingly each stage reports
``ok=None`` ("no runtime invariant; accuracy comes from the offline gate").

Like every tier it only ever touches **mutable prose spans** (code, paths, URLs, quoted text,
tool JSON, and the trailing instruction are never altered) and **fails open**: any error yields
the request unchanged. Inside **structured** messages it reduces only ``text`` blocks; immutable
blocks (tool_use/tool_result/image) are reproduced verbatim.

The actual token reduction is delegated to a :class:`TokenReducer` so the span handling is fully
testable with a fake. The production reducer (:class:`LLMLinguaReducer`) is **local** (lazy
import of the optional ``learned`` extra; the model runs locally and never makes a network call
— a compressor that phoned a remote model would defeat the project's purpose).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from parcus.model import CanonicalRequest, CompressionStats, Message, Span
from parcus.ports import TokenizerPort
from parcus.spans import classify_spans
from parcus.tokenize import default_tokenizer

__all__ = ["LLMLinguaReducer", "LearnedCompressor", "TokenReducer"]

_STEP_NAME = "learned"


@runtime_checkable
class TokenReducer(Protocol):
    """Reduces a prose string to fewer tokens while preserving meaning (local, deterministic)."""

    def reduce(self, text: str, *, keep_ratio: float) -> str:
        """Return ``text`` shortened toward ``keep_ratio`` of its tokens (best-effort)."""
        ...


def _reduce_prose(text: str, reducer: TokenReducer, keep_ratio: float) -> str:
    """Reduce only the mutable (prose) sub-spans of ``text``, preserving fenced code verbatim."""
    return "".join(
        reducer.reduce(span.text, keep_ratio=keep_ratio) if span.mutable else span.text
        for span in classify_spans(text)
    )


def _reduce_text_blocks(
    raw: dict[str, Any], reducer: TokenReducer, keep_ratio: float
) -> tuple[dict[str, Any], int]:
    """Reduce the ``text`` blocks of a structured message; other blocks stay verbatim.

    Only ``{"type": "text", "text": <str>, ...}`` blocks are touched (code-fence-aware via
    :func:`_reduce_prose`); every other block (tool_use/tool_result/image) is reproduced verbatim.
    Returns the message dict — the same object when its content isn't a block list or nothing
    changed — and the number of text blocks altered.
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
            reduced = _reduce_prose(block["text"], reducer, keep_ratio)
            if reduced != block["text"]:
                touched += 1
                block = {**block, "text": reduced}
        new_content.append(block)
    if touched == 0:
        return raw, 0
    return {**raw, "content": new_content}, touched


class LearnedCompressor:
    """Apply a local :class:`TokenReducer` to mutable spans only. Implements ``CompressorPort``.

    Args:
        reducer: The local token reducer (e.g. :class:`LLMLinguaReducer`).
        keep_ratio: Target fraction of tokens to keep (``(0, 1]``); passed to the reducer.
        tokenizer: Token counter for measurement (default heuristic).
    """

    def __init__(
        self,
        reducer: TokenReducer,
        *,
        keep_ratio: float = 0.5,
        tokenizer: TokenizerPort | None = None,
    ) -> None:
        """Initialise with the reducer, keep-ratio, and an optional token counter."""
        self._reducer = reducer
        self._keep_ratio = keep_ratio
        self._tokenizer = tokenizer or default_tokenizer()

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Return the request with mutable prose reduced, plus stats (fails open).

        Reports ``ok=None``: this tier has no model-free runtime invariant — its accuracy is
        established by the offline answer-preservation gate, not a per-request self-check.
        """
        try:
            before = self._tokenizer.count(request.text, request.model)
            touched = 0
            new_messages: list[Message] = []
            for message in request.messages:
                if message.raw is not None:
                    # Structured content: reduce text blocks; other blocks verbatim.
                    new_raw, block_touched = _reduce_text_blocks(
                        message.raw, self._reducer, self._keep_ratio
                    )
                    touched += block_touched
                    if new_raw is message.raw:
                        new_messages.append(message)
                    else:
                        new_messages.append(Message(role=message.role, spans=(), raw=new_raw))
                    continue
                new_spans: list[Span] = []
                for span in message.spans:
                    if span.mutable:
                        reduced = self._reducer.reduce(span.text, keep_ratio=self._keep_ratio)
                        if reduced != span.text:
                            touched += 1
                            new_spans.append(span.with_text(reduced))
                            continue
                    new_spans.append(span)
                new_messages.append(Message(role=message.role, spans=tuple(new_spans)))

            new_system = request.system
            if request.system is not None:
                rebuilt = _reduce_prose(request.system, self._reducer, self._keep_ratio)
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
                ok=None,  # no runtime invariant — gated offline by the answer-preservation judge
            )
            return new_request, (stats,)
        except Exception:
            # Fail open: never break the request path to save tokens.
            return request, ()


class LLMLinguaReducer:
    """Local LLMLingua-backed reducer (lazy import of the optional ``learned`` extra).

    The model is loaded from the local cache on first use and never makes a network call.

    Args:
        model_name: A local causal LM LLMLingua can drive (small by default).
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        """Defer model construction to the first :meth:`reduce` call."""
        self._model_name = model_name
        self._compressor: Any = None  # llmlingua.PromptCompressor, loaded lazily

    def reduce(self, text: str, *, keep_ratio: float) -> str:  # pragma: no cover - needs model
        """Compress ``text`` toward ``keep_ratio`` of its tokens with a local LLMLingua model."""
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise ImportError(
                    "the learned tier requires the 'learned' extra: pip install 'parcus[learned]'"
                ) from exc
            self._compressor = PromptCompressor(
                model_name=self._model_name, device_map="cpu", use_llmlingua2=False
            )
        result = self._compressor.compress_prompt(text, rate=keep_ratio)
        compressed = result["compressed_prompt"]
        return compressed if isinstance(compressed, str) else text
