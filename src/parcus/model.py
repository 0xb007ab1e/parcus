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

from dataclasses import dataclass, field, replace
from enum import StrEnum

__all__ = [
    "CachedResponse",
    "CanonicalRequest",
    "CompressionStats",
    "Dialect",
    "Message",
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
        spans: Ordered spans whose concatenation is the message body.
    """

    role: Role
    spans: tuple[Span, ...]

    @property
    def text(self) -> str:
        """The full message body (concatenation of all span texts)."""
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
    """

    dialect: Dialect
    model: str | None
    messages: tuple[Message, ...]
    system: str | None = None
    stream: bool = False
    tools_json: str | None = None

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
