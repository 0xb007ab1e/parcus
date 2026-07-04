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

from dataclasses import replace

from parcus.model import CacheCapability, CacheModel, CanonicalRequest, Dialect
from parcus.ports import CacheStrategy

__all__ = [
    "AnthropicCacheStrategy",
    "GeminiCacheStrategy",
    "NullCacheStrategy",
    "cache_strategy",
]

# Conservative Anthropic breakpoint floor: 4096 caches on every Anthropic model (Opus needs
# 4096; Sonnet-4.6/Fable cache from 2048), and up to 4 explicit breakpoints per request.
_ANTHROPIC_MIN_PREFIX_TOKENS = 4096
_ANTHROPIC_MAX_BREAKPOINTS = 4

# Conservative floor for Gemini explicit context caching. DRIFT-PRONE: Gemini's declared minimum
# cacheable prefix has changed across model generations (older 1.5 models required ~32,768 tokens;
# 2.5 models are far lower). Verify against https://ai.google.dev/gemini-api/docs/caching before
# enabling live and adjust this single constant. Below it, registering a cache is pure storage
# cost with no read discount, so the engine skips it (fail open → inline send).
_GEMINI_MIN_PREFIX_TOKENS = 4096


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
    is ``system`` + ``tools`` + every turn except the final (volatile) instruction. Injection
    (M1b) is implemented via :meth:`annotate`, which marks a breakpoint on the last stable turn;
    the dialect serialiser renders it to ``cache_control``.
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
        """Mark a cache breakpoint on the last stable turn so the prefix caches; else unchanged.

        Places the breakpoint on ``messages[boundary - 1]`` (the final protected turn), which
        caches ``tools`` + ``system`` + every turn up to and including it. Returns ``request``
        unchanged when there is no protectable message turn (``boundary`` is ``None`` or ``0`` —
        the system/tools-only case is deferred to a later slice). Fail-open: never raises.
        """
        boundary = self.cacheable_boundary(request)
        if boundary is None or boundary < 1:
            return request
        return replace(request, cache_breakpoint=boundary - 1)


class GeminiCacheStrategy:
    """Explicit context-cache policy (Gemini ``cachedContents``) — preservation only, here.

    Gemini caches a client-registered prefix behind an opaque handle managed via a **stateful,
    I/O-bearing** API (create/reference/evict + storage cost). That lifecycle deliberately does
    **not** live in this pure strategy — it belongs to a
    :class:`~parcus.ports.ContextCacheRegistrar` in the imperative shell (ADR 0010). This strategy
    contributes only the two *pure* things: the capability descriptor (so the engine knows the
    caching model + minimum prefix) and :meth:`cacheable_boundary` (the M1a guard — never let
    compression perturb the prefix that will be registered).

    :meth:`annotate` is therefore a **no-op**: referencing a context cache is a shell action
    (swap the inline prefix for a handle at serialise time), not an in-request breakpoint marker,
    so there is nothing for the pure policy to inject.
    """

    capability = CacheCapability(
        model=CacheModel.EXPLICIT_CONTEXT_API,
        min_prefix_tokens=_GEMINI_MIN_PREFIX_TOKENS,
    )

    def cacheable_boundary(self, request: CanonicalRequest) -> int | None:
        """Return the count of leading messages to protect — identical policy to Anthropic.

        The stable, re-sent prefix (``system`` + ``tools`` + all but the final volatile turn) is
        what would be registered as a context cache, so compression must not perturb it. With no
        messages, only ``system``/``tools`` could be cacheable → ``0`` if either is present, else
        ``None``.
        """
        if not request.messages:
            return 0 if (request.system or request.tools_json) else None
        return len(request.messages) - 1

    def annotate(self, request: CanonicalRequest) -> CanonicalRequest:
        """Return ``request`` unchanged — a context cache is referenced in the shell, not marked."""
        return request


_STRATEGIES: dict[Dialect, CacheStrategy] = {
    Dialect.ANTHROPIC: AnthropicCacheStrategy(),
    Dialect.GEMINI: GeminiCacheStrategy(),
}


def cache_strategy(dialect: Dialect) -> CacheStrategy:
    """Return the :class:`~parcus.ports.CacheStrategy` for ``dialect``.

    Falls back to :class:`NullCacheStrategy` for any dialect without explicit support
    (OpenAI/automatic-prefix, unknown, non-caching providers) — cache-neutral and fail-open.
    """
    return _STRATEGIES.get(dialect, NullCacheStrategy())
