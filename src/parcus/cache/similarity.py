"""Opt-in semantic (near-duplicate) response cache — serve a cached answer for a *similar* request.

The exact cache only hits on a byte-identical canonical request. A **similarity** cache widens
that: on an exact miss it finds the most similar *prior* request (by cosine over a local
embedding) and reuses its cached response when the similarity clears a high threshold. That skips
the upstream call entirely — the single largest token win — but it **trades correctness for
tokens**, so it is off by default, gated by a deliberately high threshold, and validated offline
(``parcus eval --similarity``) before an operator should raise it.

Hot-path guards (model-free, cheap) that keep it safe:

* **Threshold** — only near-duplicates qualify (default 0.97 cosine).
* **Same model** — never serve a different model's answer.
* **Same tenant** — never serve another tenant's cached response (this is threat E1 again — a
  cross-tenant similar-serve would leak data exactly like a cross-tenant exact hit).

The index stores only ``(vector, exact-key, model, tenant)`` — never prompt content — and the
*response* is fetched from the exact cache by key, so TTL/eviction and storage stay there. The
embedder is **local** (dependency-free :class:`HashingEmbedder` by default); a similarity cache
that phoned a remote embedding API would defeat the project's purpose. Every method **fails
open**: any error yields "no similar entry" (forward upstream) rather than raising.

**Embedder safety (important):** the dependency-free :class:`HashingEmbedder` is *lexical*
(bag-of-words) — it cannot distinguish requests that differ only in numbers or entities (e.g.
"scale to 10 replicas" vs "…2 replicas"), so it can produce a **false hit** even at threshold
1.0. It is therefore **not safe** for semantic caching on its own; the safe choice is the local
sentence-transformer embedder. Whichever embedder/threshold you pick, validate it with
``parcus eval --similarity`` (a no-false-hit precision gate) *before* enabling the cache — the
built-in adversarial set fails the lexical embedder by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from parcus.memory.embedding import EmbedderPort, cosine

__all__ = ["SimilarityCache", "SimilarityEntry", "SimilarityStore"]


@dataclass(frozen=True, slots=True)
class SimilarityEntry:
    """An indexed cached request: its embedding, the exact cache key, and its guards."""

    vector: list[float]
    key: str
    model: str | None
    tenant: str


@runtime_checkable
class SimilarityStore(Protocol):
    """Durable snapshot of similarity-index entries (a persistence port; fails open).

    The concrete implementation lives in :mod:`parcus.cache.similarity_store`; this protocol is
    defined here so :class:`SimilarityCache` depends only on the abstraction (no import cycle).
    """

    def load(self) -> list[SimilarityEntry]:
        """Return the persisted entries (most-recent-capped), oldest first."""
        ...

    def append(self, entry: SimilarityEntry) -> None:
        """Persist a newly indexed entry (best-effort)."""
        ...


class SimilarityCache:
    """A bounded, in-memory near-duplicate index over exact-cache keys.

    When a :class:`SimilarityStore` is supplied the index is a **snapshot**: it hydrates from the
    store at construction (so hits survive a restart) and write-throughs new entries, but every
    ``lookup`` still runs entirely in memory — persistence adds no hot-path cost.

    Args:
        embedder: Local embedder mapping request text to a vector.
        threshold: Minimum cosine similarity to count as a near-duplicate (``[0, 1]``).
        max_entries: Cap on indexed entries; oldest are evicted first (FIFO).
        store: Optional durable snapshot; hydrated on construction, written through on
            ``remember``. Any store error degrades the index to in-memory-only (fails open).
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        *,
        threshold: float = 0.97,
        max_entries: int = 2048,
        store: SimilarityStore | None = None,
    ) -> None:
        """Hold the embedder/threshold/capacity and hydrate from the store if one is given."""
        self._embedder = embedder
        self._threshold = threshold
        self._max = max_entries
        self._store = store
        self._entries: list[SimilarityEntry] = []
        if store is not None:
            try:
                self._entries = list(store.load())[-max_entries:]  # warm-start (bounded)
            except Exception:
                self._entries = []  # fail open: a broken store => empty in-memory index

    def lookup(self, *, text: str, model: str | None, tenant: str) -> str | None:
        """Return the exact-cache key of the best near-duplicate, or ``None``.

        Only entries with the **same model and tenant** are considered, and only when cosine
        similarity is ``>= threshold``. Fails open: returns ``None`` on any error.
        """
        try:
            vector = self._embedder.embed([text])[0]
            best_key: str | None = None
            best_sim = self._threshold
            for entry in self._entries:
                if entry.model != model or entry.tenant != tenant:
                    continue
                sim = cosine(vector, entry.vector)
                if sim >= best_sim:
                    best_sim = sim
                    best_key = entry.key
            return best_key
        except Exception:
            return None

    def remember(self, *, text: str, key: str, model: str | None, tenant: str) -> None:
        """Index a freshly cached request for future near-duplicate lookups (fails open).

        When a store is configured the entry is also written through to it (best-effort); a store
        failure leaves the in-memory index intact — persistence never breaks the request path.
        """
        try:
            vector = self._embedder.embed([text])[0]
            entry = SimilarityEntry(vector=vector, key=key, model=model, tenant=tenant)
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries.pop(0)  # FIFO eviction — a cap, not a correctness guarantee
        except Exception:
            return
        if self._store is not None:
            try:
                self._store.append(entry)  # write-through snapshot (fails open)
            except Exception:
                return
