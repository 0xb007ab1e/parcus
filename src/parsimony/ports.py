"""Ports (interfaces) for the functional core.

These are :class:`typing.Protocol` definitions — structural interfaces the pure core depends
on. Concrete adapters (httpx upstream, SQLite store, tiktoken tokenizer, optional local
models) implement them and are injected at the composition root, never imported by the core.
See ``docs/adr/0001-proxy-architecture-and-fail-open.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parsimony.model import (
    CachedResponse,
    CanonicalRequest,
    CompressionStats,
    RedactionReport,
)

__all__ = [
    "CachePort",
    "ClockPort",
    "CompressorPort",
    "MemoryPort",
    "RedactorPort",
    "TokenizerPort",
]


@runtime_checkable
class TokenizerPort(Protocol):
    """Counts tokens for a given model so savings can be measured exactly."""

    def count(self, text: str, model: str | None = None) -> int:
        """Return the number of tokens ``text`` encodes to for ``model``.

        Implementations must be deterministic and side-effect-free. A heuristic fallback is
        acceptable when the exact provider tokenizer is unavailable, provided it is stable.
        """
        ...


@runtime_checkable
class CompressorPort(Protocol):
    """Transforms a canonical request to use fewer tokens, preserving meaning."""

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Return a (possibly) reduced request plus per-pass statistics.

        Implementations MUST be lossless with respect to immutable spans and MUST fail open:
        on any internal error they return the input request unchanged with empty stats rather
        than raising.
        """
        ...


@runtime_checkable
class ClockPort(Protocol):
    """Injected time source (for TTLs and testability)."""

    def now(self) -> float:
        """Return the current time as a Unix timestamp (seconds)."""
        ...


@runtime_checkable
class RedactorPort(Protocol):
    """Masks secrets/PII in text before it is persisted or logged.

    Never applied to the request forwarded upstream nor to a replayed cache response (those
    must stay verbatim); only to derived/stored/logged content.
    """

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        """Return ``text`` with sensitive spans masked, plus a report of what matched."""
        ...

    def has_secret(self, text: str) -> bool:
        """Return whether ``text`` contains any detectable secret (for the no-cache bypass)."""
        ...


@runtime_checkable
class CachePort(Protocol):
    """An exact-match response cache keyed by a hash of the canonical request.

    Implementations MUST fail open: a get/put error returns ``None``/no-ops rather than
    raising, since the cache is a performance layer and the system must be correct when the
    cache is empty or unavailable.
    """

    def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for ``key`` if present and unexpired, else ``None``."""
        ...

    def put(self, key: str, value: CachedResponse, ttl_seconds: int) -> None:
        """Store ``value`` under ``key`` with a time-to-live in seconds."""
        ...


@runtime_checkable
class MemoryPort(Protocol):
    """Read-augmenting memory over prior context (graph-backed).

    ``ingest`` records a request's content; ``relevant`` retrieves the most relevant prior
    snippets for a query. Used (in a later, eval-gated slice) to inject only the relevant
    subgraph instead of re-sending large context, and for conversation compaction.
    """

    def ingest(self, request: CanonicalRequest) -> None:
        """Record the request's content into memory."""
        ...

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        """Return up to ``limit`` prior snippets most relevant to ``query``."""
        ...
