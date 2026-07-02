"""Canonical, provider-agnostic request model and transform statistics.

Each provider dialect (Anthropic Messages, OpenAI Chat Completions, …) is parsed into a
single :class:`CanonicalRequest` on the way in and re-serialised to its original dialect on
the way out. The pure transform/cache logic operates only on this model, so it never depends
on a provider SDK. Anything that cannot be parsed is left as ``None`` and passed through
untouched (fail open).

Spans carry a ``mutable`` flag: compression passes may only touch *mutable* spans. Code,
file paths, URLs, quoted strings, tool JSON, numbers/IDs, and the trailing user instruction
are immutable by construction (see :mod:`parcus.compress`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

__all__ = [
    "CacheCapability",
    "CacheModel",
    "CachedResponse",
    "CanonicalRequest",
    "CompressionStats",
    "Dialect",
    "Message",
    "ProviderUsage",
    "RedactionReport",
    "Role",
    "Span",
]


class Role(StrEnum):
    """The author of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Dialect(StrEnum):
    """The provider wire format a request arrived in."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    UNKNOWN = "unknown"


class CacheModel(StrEnum):
    """How a provider caches re-sent request prefixes — drives the per-provider cache strategy.

    * ``NONE`` — the provider does not cache prefixes (e.g. Groq). Nothing to preserve; the whole
      request may be compressed.
    * ``AUTOMATIC_PREFIX`` — the provider caches a stable prefix on its own with no client control
      (e.g. OpenAI, DeepSeek). parcus can only *preserve* it (not perturb the cacheable prefix).
    * ``EXPLICIT_BREAKPOINT`` — the provider caches at client-declared breakpoints (Anthropic
      ``cache_control``). parcus can both *preserve* and *inject* a breakpoint.
    """

    NONE = "none"
    AUTOMATIC_PREFIX = "automatic_prefix"
    EXPLICIT_BREAKPOINT = "explicit_breakpoint"


@dataclass(frozen=True, slots=True)
class CacheCapability:
    """A provider's prompt-cache capabilities, consulted by the engine when it wires in caching.

    This is plain data (deliberately not a generic type parameterised on the provider): the
    core stays provider-blind and each dialect's strategy carries its own descriptor. See
    ``docs/design/token-reduction-roadmap.md`` §2.1.

    Args:
        model: The provider's caching model.
        min_prefix_tokens: The smallest prefix the provider will cache (0 when not applicable).
            For Anthropic this is the conservative floor across models (4096; Sonnet-4.6/Fable
            cache from 2048) so an injected breakpoint is guaranteed to cache on any model.
        max_breakpoints: The maximum number of explicit cache breakpoints per request (0 when
            the provider takes no explicit breakpoints).
    """

    model: CacheModel
    min_prefix_tokens: int = 0
    max_breakpoints: int = 0


@dataclass(frozen=True, slots=True)
class Span:
    """A contiguous run of message text with a mutability flag.

    Args:
        text: The literal text of the span.
        mutable: Whether compression passes are permitted to alter this span. Immutable
            spans (code, paths, URLs, quoted text, tool JSON, the trailing instruction) are
            reproduced verbatim.
    """

    text: str
    mutable: bool = True

    def with_text(self, text: str) -> Span:
        """Return a copy of this span with replaced text, preserving ``mutable``."""
        return replace(self, text=text)


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation turn decomposed into spans.

    Args:
        role: Who authored the message.
        spans: Ordered spans whose concatenation is the message body (empty for a structured
            message carried via ``raw``).
        raw: For **structured** content (tool_use/tool_result/image blocks, OpenAI tool calls)
            that the text-only path can't decompose, the verbatim original message dict — so the
            serializer reproduces it byte-for-byte and optimizations leave it untouched. ``None``
            for a plain-text message (the common case). See
            ``docs/design/structured-content-parser.md``.
    """

    role: Role
    spans: tuple[Span, ...]
    raw: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        """The full message body — span concatenation, or a stable JSON view of ``raw``.

        A structured message's stable JSON view keeps the cache key (which hashes ``text``)
        unique across differing structured content and gives token measurement something to
        count. It is not the wire form — the serializer writes ``raw`` verbatim.
        """
        if self.raw is not None:
            return json.dumps(self.raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "".join(s.text for s in self.spans)


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    """A provider-agnostic view of an inference request.

    Args:
        dialect: The wire format the request arrived in.
        model: The model identifier requested, if known.
        system: The system prompt, if any (kept separate as providers do).
        messages: The conversation turns.
        stream: Whether the client requested a streaming response.
        tools_json: Verbatim, immutable tool/function schema JSON, if present.
        cache_breakpoint: Index into ``messages`` of the message whose content-end carries an
            abstract prompt-cache breakpoint, or ``None`` for none. Set by a
            :class:`~parcus.ports.CacheStrategy`'s ``annotate`` (provider-agnostic policy); the
            dialect serialiser renders it to the provider's concrete form (Anthropic
            ``cache_control``) or ignores it. See ``docs/design/token-reduction-roadmap.md`` §2.1.
    """

    dialect: Dialect
    model: str | None
    messages: tuple[Message, ...]
    system: str | None = None
    stream: bool = False
    tools_json: str | None = None
    cache_breakpoint: int | None = None

    @property
    def text(self) -> str:
        """All mutable+immutable text concatenated (for measurement only)."""
        head = self.system or ""
        return head + "".join(m.text for m in self.messages)


@dataclass(frozen=True, slots=True)
class CompressionStats:
    """Per-pass measurement emitted by every compression step.

    Args:
        step: Identifier of the compression step that produced these stats.
        tokens_before: Token count of the input.
        tokens_after: Token count of the output.
        spans_touched: Number of mutable spans actually modified.
        notes: Optional human-readable detail for audit/eval.
        ok: Result of the pass's model-free self-check (its invariant held), or ``None`` if
            the pass does not have a runtime invariant. Drives the live accuracy metric.
    """

    step: str
    tokens_before: int
    tokens_after: int
    spans_touched: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)
    ok: bool | None = None

    @property
    def tokens_saved(self) -> int:
        """Tokens removed by this pass (never negative in practice; clamped at 0)."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def ratio(self) -> float:
        """Fraction of input tokens removed, in ``[0.0, 1.0]``.

        Returns ``0.0`` when the input had no tokens (avoids division by zero).
        """
        if self.tokens_before <= 0:
            return 0.0
        return self.tokens_saved / self.tokens_before


@dataclass(frozen=True, slots=True)
class RedactionReport:
    """The outcome of scanning text for secrets/PII before it is persisted or logged.

    Args:
        total: Total number of spans masked.
        categories: Sorted, de-duplicated category names that matched (e.g. ``"api_key"``).
    """

    total: int
    categories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_secrets(self) -> bool:
        """Whether any sensitive span was detected."""
        return self.total > 0


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    """The provider's **own** reported token usage, parsed from a forwarded response.

    This is ground truth — the billed counts — as opposed to parcus's local-tokenizer estimate
    of the request. It also exposes the provider's **prompt-cache** behaviour, which is the
    dominant cost lever for tool/history-heavy harnesses: ``cache_read_tokens`` are re-sent prefix
    tokens the provider served from its cache (cheap), and ``cache_write_tokens`` were written to
    it. Watching these confirms whether parcus's request compression *preserved* the provider's
    cache hit (good) or *perturbed the cacheable prefix and busted it* (a net-negative regression
    — PLAN research Q3). All counts default to 0 when the provider doesn't report them.

    Args:
        input_tokens: Billed input/prompt tokens.
        output_tokens: Billed output/completion tokens.
        cache_read_tokens: Input tokens served from the provider's prompt cache.
        cache_write_tokens: Input tokens written to the provider's prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """A provider response stored for exact-match replay.

    The body is stored **verbatim** so a cache hit replays the original result byte-for-byte
    (responses are never modified). Treat the store as confidential (see the threat model).

    Args:
        status_code: HTTP status of the original response.
        body: The raw response body bytes.
        content_type: The original ``Content-Type``, if known.
    """

    status_code: int
    body: bytes
    content_type: str | None = None
