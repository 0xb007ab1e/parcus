"""Exact-match response cache (Track A of the graph-memory plan).

By default the cache hits only on an *identical* (normalised) request — the safest policy for
stateful agentic loops. Components:

* :func:`~parcus.cache.key.compute_key` — deterministic, salted hash of the canonical
  request (prompts are not stored, only the hash);
* :class:`~parcus.cache.policy.CachePolicy` — whether a request may be cached at all
  (no-cache patterns + credential bypass);
* :class:`~parcus.cache.sqlite_cache.SqliteCache` — the confidential, TTL-bound store;
* :class:`~parcus.cache.clock.SystemClock` — injected time source.

Embedding-similarity reuse (:class:`~parcus.cache.similarity.SimilarityCache`) is a separate,
**off-by-default** mode that widens hits to near-duplicate requests — see its module docstring.
"""

from parcus.cache.clock import SystemClock
from parcus.cache.key import compute_key
from parcus.cache.null import NullCache
from parcus.cache.policy import CachePolicy
from parcus.cache.similarity import SimilarityCache, SimilarityEntry, SimilarityStore
from parcus.cache.similarity_store import SqliteSimilarityStore
from parcus.cache.sqlite_cache import SqliteCache

__all__ = [
    "CachePolicy",
    "NullCache",
    "SimilarityCache",
    "SimilarityEntry",
    "SimilarityStore",
    "SqliteCache",
    "SqliteSimilarityStore",
    "SystemClock",
    "compute_key",
]
