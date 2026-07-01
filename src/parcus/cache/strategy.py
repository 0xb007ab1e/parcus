"""Per-provider prompt-cache strategies + a dialect-keyed registry.

Concrete :class:`~parcus.ports.CacheStrategy` adapters, one per provider caching model, plus
:func:`cache_strategy` which resolves a :class:`~parcus.model.Dialect` to its strategy and
falls back to :class:`NullCacheStrategy` (cache-neutral, fail-open) for any dialect without
explicit support. This is the first slice of the token-reduction roadmap's M1a
(cache-preservation); breakpoint *injection* (M1b) renders in the dialect serialiser and is not
wired here yet. See ``docs/design/token-reduction-roadmap.md`` §2.1.

The strategies are pure and deterministic — no I/O, no tokenizer — so the engine (which owns
the tokenizer) enforces :attr:`~parcus.model.CacheCapability.min_prefix_tokens` when it consults
a boundary. Nothing here modifies a provider response or emits provider wire JSON.
"""

from __future__ import annotations

from parcus.model import CacheCapability, CacheModel, CanonicalRequest, Dialect
from parcus.ports import CacheStrategy

__all__ = [
    "AnthropicCacheStrategy",
    "NullCacheStrategy",
    "cache_strategy",
]

# Conservative Anthropic breakpoint floor: 4096 caches on every Anthropic model (Opus needs
# 4096; Sonnet-4.6/Fable cache from 2048), and up to 4 explicit breakpoints per request.
_ANTHROPIC_MIN_PREFIX_TOKENS = 4096
_ANTHROPIC_MAX_BREAKPOINTS = 4


class NullCacheStrategy:
    """The cache-neutral default: no prefix is protected and nothing is injected.

    The fail-open choice for providers with no prompt caching (e.g. Groq) and for any unknown
    dialect — it makes "cache-neutral-or-better on every provider" a structural default rather
    than a matter of discipline. Compression may touch the whole request.
    """

    capability = CacheCapability(model=CacheModel.NONE)

    def cacheable_boundary(self, request: CanonicalRequest) -> int | None:
        """Return ``None`` — nothing is cacheable, so the whole request may be compressed."""
        return None

    def annotate(self, request: CanonicalRequest) -> CanonicalRequest:
        """Return ``request`` unchanged — a non-caching provider has no breakpoint to inject."""
        return request


class AnthropicCacheStrategy:
    """Explicit-breakpoint caching (Anthropic ``cache_control``).

    Preservation (M1a) is implemented via :meth:`cacheable_boundary`. The stable, re-sent prefix
    is ``system`` + ``tools`` + every turn except the final (volatile) instruction. Breakpoint
    *injection* (M1b) requires the dialect serialiser to render ``cache_control`` and is deferred
    to that slice, so :meth:`annotate` is currently the identity (a safe, fail-open no-op).
    """

    capability = CacheCapability(
        model=CacheModel.EXPLICIT_BREAKPOINT,
        min_prefix_tokens=_ANTHROPIC_MIN_PREFIX_TOKENS,
        max_breakpoints=_ANTHROPIC_MAX_BREAKPOINTS,
    )

    def cacheable_boundary(self, request: CanonicalRequest) -> int | None:
        """Return the count of leading messages to protect (all but the final turn).

        With messages present, everything except the last turn is stable → ``len - 1``. With no
        messages, only ``system``/``tools`` could be cacheable → ``0`` if either is present, else
        ``None`` (nothing to protect).
        """
        if not request.messages:
            return 0 if (request.system or request.tools_json) else None
        return len(request.messages) - 1

    def annotate(self, request: CanonicalRequest) -> CanonicalRequest:
        """Return ``request`` unchanged — breakpoint injection (M1b) is a later serialiser slice."""
        return request


_STRATEGIES: dict[Dialect, CacheStrategy] = {
    Dialect.ANTHROPIC: AnthropicCacheStrategy(),
}


def cache_strategy(dialect: Dialect) -> CacheStrategy:
    """Return the :class:`~parcus.ports.CacheStrategy` for ``dialect``.

    Falls back to :class:`NullCacheStrategy` for any dialect without explicit support
    (OpenAI/automatic-prefix, unknown, non-caching providers) — cache-neutral and fail-open.
    """
    return _STRATEGIES.get(dialect, NullCacheStrategy())
