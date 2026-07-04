"""Ports (interfaces) for the functional core.

These are :class:`typing.Protocol` definitions — structural interfaces the pure core depends
on. Concrete adapters (httpx upstream, SQLite store, tiktoken tokenizer, optional local
models) implement them and are injected at the composition root, never imported by the core.
See ``docs/adr/0001-proxy-architecture-and-fail-open.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parcus.model import (
    CacheCapability,
    CachedResponse,
    CanonicalRequest,
    CompressionStats,
    ContextCacheHandle,
    RedactionReport,
)

__all__ = [
    "CachePort",
    "CacheStrategy",
    "ClockPort",
    "CompressorPort",
    "ContextCacheRegistrar",
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

    def get(self, key: str, *, tenant: str = "") -> CachedResponse | None:
        """Return the cached response for ``key`` if present and unexpired, else ``None``.

        ``tenant`` is the credential-derived tenant id (empty in single-tenant mode); plain stores
        ignore it, while an encrypting cache uses it to select the per-tenant key.
        """
        ...

    def put(self, key: str, value: CachedResponse, ttl_seconds: int, *, tenant: str = "") -> None:
        """Store ``value`` under ``key`` with a time-to-live in seconds (``tenant`` as in get)."""
        ...


@runtime_checkable
class CacheStrategy(Protocol):
    """Provider-specific prompt-cache policy — a uniform port with one adapter per dialect.

    Extracts the per-provider divergence in prompt caching (Anthropic ``cache_control`` vs
    OpenAI automatic-prefix vs none) behind one interface so the provider-blind core adapts via
    a ``Dialect``-keyed registry rather than a type generic over the provider. Implementations
    are **pure and deterministic** (no I/O, no tokenizer) and never emit provider wire JSON —
    that rendering belongs to the dialect serialiser (policy vs representation). See
    ``docs/design/token-reduction-roadmap.md`` §2.1.
    """

    capability: CacheCapability

    def cacheable_boundary(self, request: CanonicalRequest) -> int | None:
        """Return the count of leading messages in the provider-cacheable, must-not-perturb prefix.

        ``system`` and ``tools`` are implicitly part of that prefix whenever a value is returned.
        Compression may then touch only ``messages[boundary:]`` (the volatile tail). ``None``
        means there is no worthwhile cacheable prefix and the whole request may be compressed —
        always the case for a non-caching provider. This feeds the M1a cache-preservation guard;
        the engine additionally enforces :attr:`CacheCapability.min_prefix_tokens` (it owns the
        tokenizer). Must be side-effect-free.
        """
        ...

    def annotate(self, request: CanonicalRequest) -> CanonicalRequest:
        """Return ``request`` with a cache breakpoint marked, or unchanged if injection is moot.

        A no-op for ``NONE``/``AUTOMATIC_PREFIX`` providers (nothing to inject). For an
        explicit-breakpoint provider this is the M1b injection hook; rendering the marker to
        ``cache_control`` is the dialect serialiser's job. Fail-open: returning the request
        unchanged is always safe.
        """
        ...


@runtime_checkable
class ContextCacheRegistrar(Protocol):
    """Stateful lifecycle for a provider **explicit context cache** (Gemini ``cachedContents``).

    Unlike :class:`CacheStrategy` — a *pure* request→request policy — this port does **network
    I/O** and holds **handle state**: it registers a large stable prefix with the provider,
    returns a reusable :class:`~parcus.model.ContextCacheHandle`, and evicts expired ones to bound
    the per-hour storage cost. It therefore lives in the imperative shell and is injected; the pure
    core never calls a provider client directly (ADR 0001 / ADR 0010).

    Implementations MUST **fail open**: :meth:`ensure` returns ``None`` (⇒ forward the prefix
    inline, uncached) on any error, a below-worthwhile prefix, a spend-cap hit, or a miss it can't
    create; :meth:`evict_expired` no-ops on error. Handles MUST be scoped per ``(tenant, model,
    prefix)`` so a cache is never shared across tenants or models (a handle is only valid for the
    credential/project that created it). Since this changes only billing/transport and never
    request or response content, it needs no answer-preservation gate.
    """

    def ensure(
        self, prefix: str, *, model: str | None, tenant: str = ""
    ) -> ContextCacheHandle | None:
        """Return a live handle for ``prefix`` under ``(tenant, model)``, creating one if useful.

        ``None`` means "not cached — forward the prefix inline," always the safe, fail-open answer
        (a fresh, expired, spend-capped, or errored lookup all resolve to ``None``). Side-effectful
        (may call the provider and persist the handle) but MUST NOT raise.
        """
        ...

    def evict_expired(self) -> None:
        """Delete provider caches whose tracked TTL has lapsed to bound cost; never raises."""
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
