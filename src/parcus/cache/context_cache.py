"""Context-cache registrars — the stateful, I/O-bearing side of ``EXPLICIT_CONTEXT_API`` caching.

Concrete :class:`~parcus.ports.ContextCacheRegistrar` adapters for providers that cache a
client-registered prefix behind an opaque handle (Gemini ``cachedContents``): register a large
stable prefix once, reference the handle on later turns for a token discount, and evict expired
handles to bound the per-hour storage cost. See ``docs/adr/0010-gemini-context-cache-adapter.md``.

Design (ADR 0010):

* :class:`NullContextCacheRegistrar` is the **fail-open default** — it caches nothing and is the
  registrar wired for every provider without explicit context caching (and whenever the feature
  is off). ``ensure`` always returns ``None`` ⇒ the engine forwards the prefix inline.
* :class:`GeminiContextCacheRegistrar` owns the **pure lifecycle policy** (reuse a live handle,
  treat an expired one as a miss, respect a spend cap, fail open on any error). The two genuine
  I/O operations — creating and deleting a provider cache — are **injected callables**, so the
  policy is fully testable with fakes and only the real SDK wiring (:func:`gemini_registrar`) is
  provider/network-dependent (``# pragma: no cover``).

Nothing here touches a request/response body — context caching changes only billing and transport
— so, unlike the Tier-2 learned compressor, it needs no answer-preservation gate.

**Not yet on the request path.** Routing a Gemini request (detect ``generateContent`` → parse →
serialise a ``cachedContent`` reference) and wiring a registrar into the engine's forward path are
the follow-up slice; this module lands the seams.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable

from parcus.model import ContextCacheHandle
from parcus.ports import ClockPort

__all__ = [
    "GeminiContextCacheRegistrar",
    "NullContextCacheRegistrar",
    "gemini_registrar",
]


class NullContextCacheRegistrar:
    """The cache-neutral default: registers nothing, holds no state, spends nothing.

    The fail-open choice for any provider without explicit context caching and whenever the
    feature is disabled — it makes "no context cache unless one is deliberately wired" a structural
    default rather than a matter of discipline.
    """

    def ensure(
        self, prefix: str, *, model: str | None, tenant: str = ""
    ) -> ContextCacheHandle | None:
        """Return ``None`` — nothing is cached, so the caller forwards the prefix inline."""
        return None

    def evict_expired(self) -> None:
        """Do nothing — there is no state to evict."""
        return None


class GeminiContextCacheRegistrar:
    """Lifecycle policy for Gemini ``cachedContents``, with I/O injected for testability.

    Holds live handles keyed by ``(tenant, model, sha256(prefix))`` so a cache is never shared
    across tenants or models. ``ensure`` reuses a non-expired handle, treats an expired one as a
    miss, refuses to create past the spend cap, and **fails open** (returns ``None``) on any error
    — the worst case is "no discount this turn, send inline." The provider remains the ultimate
    authority on TTL; this tracks expiry conservatively so it never references a dropped cache.

    Args:
        clock: Injected time source (Unix seconds) for TTL bookkeeping and testability.
        ttl_seconds: How long a freshly-created handle is considered live (``<=`` the provider TTL).
        max_entries: Spend cap — the maximum number of concurrently *live* provider caches. At the
            cap, a miss fails open instead of registering another paid resource.
        create: Injected I/O — register ``prefix`` for ``model`` with the provider, returning the
            opaque resource name. May raise; the caller catches and fails open.
        delete: Injected I/O — delete a provider cache by resource name. May raise; ignored.
    """

    def __init__(
        self,
        *,
        clock: ClockPort,
        ttl_seconds: int,
        max_entries: int,
        create: Callable[[str, str | None], str],
        delete: Callable[[str], None],
    ) -> None:
        """Store the injected clock, TTL/cap policy, and provider create/delete I/O callables."""
        self._clock = clock
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._create = create
        self._delete = delete
        self._handles: dict[tuple[str, str, str], ContextCacheHandle] = {}

    @staticmethod
    def _key(prefix: str, model: str | None, tenant: str) -> tuple[str, str, str]:
        """Return the per-``(tenant, model, prefix)`` store key (prefix hashed, not stored raw)."""
        digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        return (tenant, model or "", digest)

    def _live_count(self, now: float) -> int:
        """Number of currently non-expired handles (what the spend cap bounds)."""
        return sum(1 for h in self._handles.values() if not h.is_expired(now))

    def ensure(
        self, prefix: str, *, model: str | None, tenant: str = ""
    ) -> ContextCacheHandle | None:
        """Return a live handle for ``prefix``, creating one if useful; ``None`` ⇒ send inline."""
        try:
            key = self._key(prefix, model, tenant)
            now = self._clock.now()
            existing = self._handles.get(key)
            if existing is not None and not existing.is_expired(now):
                return existing
            # Miss or stale: drop the stale entry, honour the spend cap, then register.
            self._handles.pop(key, None)
            if self._live_count(now) >= self._max_entries:
                return None  # spend cap reached — don't pay for another cache
            name = self._create(prefix, model)
            handle = ContextCacheHandle(
                name=name, model=model or "", expires_at=now + self._ttl_seconds
            )
            self._handles[key] = handle
            return handle
        except Exception:
            return None  # fail open: any error ⇒ forward the prefix inline

    def evict_expired(self) -> None:
        """Delete every handle whose tracked TTL has lapsed; per-handle delete errors are ignored.

        Provider deletion is best-effort — a failed delete leaves a paid resource to lapse on the
        provider's own TTL, never breaking the caller.
        """
        now = self._clock.now()
        for key, handle in list(self._handles.items()):
            if handle.is_expired(now):
                self._handles.pop(key, None)
                with contextlib.suppress(Exception):
                    self._delete(handle.name)


def gemini_registrar(
    *, clock: ClockPort, ttl_seconds: int, max_entries: int, api_key: str
) -> GeminiContextCacheRegistrar:  # pragma: no cover - needs the `gemini` extra + a live key
    """Build a :class:`GeminiContextCacheRegistrar` wired to the real ``google-genai`` client.

    Lazy-imports the optional ``gemini`` extra so the dependency is only required when the feature
    is actually enabled; if it is absent, this raises :class:`ImportError` and the composition root
    falls back to :class:`NullContextCacheRegistrar` (feature off, proxy unaffected). The provider
    calls are `# pragma: no cover` — exercised only with the extra installed and a live API key,
    never in hermetic CI. The pure lifecycle policy above is what the test suite drives.
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    def create(prefix: str, model: str | None) -> str:
        cached = client.caches.create(
            model=model or "gemini-2.5-flash",
            config={"contents": prefix},
        )
        return str(cached.name)

    def delete(name: str) -> None:
        client.caches.delete(name=name)

    return GeminiContextCacheRegistrar(
        clock=clock,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        create=create,
        delete=delete,
    )
