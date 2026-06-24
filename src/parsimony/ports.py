"""Ports (interfaces) for the functional core.

These are :class:`typing.Protocol` definitions — structural interfaces the pure core depends
on. Concrete adapters (httpx upstream, SQLite store, tiktoken tokenizer, optional local
models) implement them and are injected at the composition root, never imported by the core.
See ``docs/adr/0001-proxy-architecture-and-fail-open.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parsimony.model import CanonicalRequest, CompressionStats

__all__ = ["ClockPort", "CompressorPort", "TokenizerPort"]


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
