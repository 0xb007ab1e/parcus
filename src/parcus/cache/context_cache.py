"""Context-cache registrars — the stateful, I/O-bearing side of ``EXPLICIT_CONTEXT_API`` caching.

Concrete :class:`~parcus.ports.ContextCacheRegistrar` adapters for providers that cache a
client-registered prefix behind an opaque handle (Gemini ``cachedContents``): register a large
stable prefix once, reference the handle on later turns for a token discount, and drop stale
handles to bound memory. See ``docs/adr/0010-gemini-context-cache-adapter.md``.

Design (ADR 0010, Updates 2026-07-04 / 2026-07-07):

* :class:`NullContextCacheRegistrar` is the **fail-open default** — it caches nothing and is the
  registrar wired for every provider without explicit context caching (and whenever the feature
  is off). ``ensure`` always returns ``None`` ⇒ the engine forwards the prefix inline.
* :class:`GeminiContextCacheRegistrar` owns the **pure lifecycle policy** (reuse a live handle,
  treat an expired one as a miss, respect a spend cap, prune expired entries, fail open on any
  error). The one genuine I/O operation — creating a provider cache — is an **injected async
  callable**, so the policy is fully testable with fakes and only the real SDK wiring
  (:func:`gemini_registrar`) is provider/network-dependent (``# pragma: no cover``).

**Credential-scoped, per request.** A Gemini ``cachedContents`` handle is only valid for the API
key/project that created it. parcus is a provider-blind proxy — it forwards the *caller's*
``x-goog-api-key`` — so ``create`` runs under the **caller's** credential (passed per request) and
handles are keyed by a **fingerprint of that credential** (``sha256``), never the raw key. This
isolates handles across callers even in single-tenant mode (where the tenant id is empty), so a
caller can never reference a cache another credential owns. The raw key is used transiently for
the ``create`` call and is **never stored in a handle or logged**.

**No remote delete.** Eviction only prunes the local handle map; the remote provider cache lapses
on the provider's own TTL (the registrar's tracked TTL is ``<=`` that), so eviction needs no
credential. A dropped-but-not-yet-expired remote cache simply lingers until the provider expires
it — the spend cap bounds how many we *create* concurrently.

Nothing here touches a request/response body — context caching changes only billing/transport —
so, unlike the Tier-2 learned compressor, it needs no answer-preservation gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

from parcus.model import ContextCacheHandle
from parcus.ports import ClockPort

__all__ = [
    "GeminiContextCacheRegistrar",
    "NullContextCacheRegistrar",
    "gemini_registrar",
]

# Prune the handle map opportunistically once it exceeds this multiple of the spend cap, so
# in-memory growth is bounded structurally rather than only by an external eviction schedule.
_PRUNE_FACTOR = 4


class NullContextCacheRegistrar:
    """The cache-neutral default: registers nothing, holds no state, spends nothing.

    The fail-open choice for any provider without explicit context caching and whenever the
    feature is disabled — it makes "no context cache unless one is deliberately wired" a structural
    default rather than a matter of discipline.
    """

    async def ensure(
        self, prefix: str, *, model: str | None, credential: str
    ) -> ContextCacheHandle | None:
        """Return ``None`` — nothing is cached, so the caller forwards the prefix inline."""
        return None

    async def evict_expired(self) -> None:
        """Do nothing — there is no state to evict."""
        return None


class GeminiContextCacheRegistrar:
    """Lifecycle policy for Gemini ``cachedContents``, with the create I/O injected for testing.

    Holds live handles keyed by ``(sha256(credential), model, sha256(prefix))`` so a cache is
    never shared across credentials or models. ``ensure`` reuses a non-expired handle, treats an
    expired one as a miss, refuses to create past the spend cap, prunes expired entries, and
    **fails open** (returns ``None``) on any error — the worst case is "no discount this turn, send
    inline." The provider remains the ultimate authority on TTL; this tracks expiry conservatively
    so it never references a dropped cache.

    Args:
        clock: Injected time source (Unix seconds) for TTL bookkeeping and testability.
        ttl_seconds: How long a freshly-created handle is considered live (``<=`` the provider TTL).
        max_entries: Spend cap — the maximum number of concurrently *live* provider caches. At the
            cap, a miss fails open instead of registering another paid resource.
        create: Injected I/O — register ``prefix`` for ``model`` under the caller's ``credential``
            with the provider, returning the opaque resource name. May raise; the caller catches
            and fails open.
    """

    def __init__(
        self,
        *,
        clock: ClockPort,
        ttl_seconds: int,
        max_entries: int,
        create: Callable[[str, str | None, str], Awaitable[str]],
    ) -> None:
        """Store the injected clock, TTL/cap policy, and the provider ``create`` I/O callable."""
        self._clock = clock
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._create = create
        self._handles: dict[tuple[str, str, str], ContextCacheHandle] = {}

    @staticmethod
    def _key(credential: str, model: str | None, prefix: str) -> tuple[str, str, str]:
        """Return the per-``(credential, model, prefix)`` key (credential + prefix both hashed)."""
        cred = hashlib.sha256(credential.encode("utf-8")).hexdigest()
        digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        return (cred, model or "", digest)

    def _live_count(self, now: float) -> int:
        """Number of currently non-expired handles (what the spend cap bounds)."""
        return sum(1 for h in self._handles.values() if not h.is_expired(now))

    def _prune(self, now: float) -> None:
        """Drop expired handles from the map; no remote call (the provider TTL owns the cache)."""
        for key in [k for k, h in self._handles.items() if h.is_expired(now)]:
            del self._handles[key]

    async def ensure(
        self, prefix: str, *, model: str | None, credential: str
    ) -> ContextCacheHandle | None:
        """Return a live handle for ``prefix`` under ``credential``; ``None`` ⇒ send inline."""
        try:
            now = self._clock.now()
            if len(self._handles) > self._max_entries * _PRUNE_FACTOR:
                self._prune(now)  # opportunistic: keep the map bounded without a scheduler
            key = self._key(credential, model, prefix)
            existing = self._handles.get(key)
            if existing is not None and not existing.is_expired(now):
                return existing
            self._handles.pop(key, None)  # a stale same-key handle → recreate
            if self._live_count(now) >= self._max_entries:
                return None  # spend cap reached — don't pay for another cache
            name = await self._create(prefix, model, credential)
            handle = ContextCacheHandle(
                name=name, model=model or "", expires_at=now + self._ttl_seconds
            )
            self._handles[key] = handle
            return handle
        except Exception:
            return None  # fail open: any error ⇒ forward the prefix inline

    async def evict_expired(self) -> None:
        """Prune expired handles from the local map; the remote cache lapses on the provider TTL."""
        self._prune(self._clock.now())


def gemini_registrar(
    *, clock: ClockPort, ttl_seconds: int, max_entries: int
) -> GeminiContextCacheRegistrar:  # pragma: no cover - needs the `gemini` extra + a live key
    """Build a :class:`GeminiContextCacheRegistrar` wired to the real ``google-genai`` client.

    ``create`` builds a client from the **caller's** credential per call (handles are key-scoped,
    and this avoids retaining any key), lazy-importing the optional ``gemini`` extra so the
    dependency is only required when the feature is enabled; if it is absent, ``create`` raises and
    ``ensure`` fails open (feature effectively off, proxy unaffected). The provider call is
    ``# pragma: no cover`` — exercised only with the extra installed and a live API key, never in
    hermetic CI. The pure lifecycle policy above is what the test suite drives.
    """

    async def create(prefix: str, model: str | None, credential: str) -> str:
        from google import genai

        client = genai.Client(api_key=credential)
        cached = await client.aio.caches.create(
            model=model or "gemini-2.5-flash",
            config={"contents": prefix},
        )
        return str(cached.name)

    return GeminiContextCacheRegistrar(
        clock=clock,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        create=create,
    )
