"""Exact-match response cache (Track A of the graph-memory plan).

By default the cache hits only on an *identical* (normalised) request — the safest policy for
stateful agentic loops. Components:

* :func:`~parsimony.cache.key.compute_key` — deterministic, salted hash of the canonical
  request (prompts are not stored, only the hash);
* :class:`~parsimony.cache.policy.CachePolicy` — whether a request may be cached at all
  (no-cache patterns + credential bypass);
* :class:`~parsimony.cache.sqlite_cache.SqliteCache` — the confidential, TTL-bound store;
* :class:`~parsimony.cache.clock.SystemClock` — injected time source.

Embedding-similarity reuse is deliberately **not** part of this module; it is a separate,
off-by-default mode planned for a later milestone.
"""

from parsimony.cache.clock import SystemClock
from parsimony.cache.key import compute_key
from parsimony.cache.null import NullCache
from parsimony.cache.policy import CachePolicy
from parsimony.cache.sqlite_cache import SqliteCache

__all__ = ["CachePolicy", "NullCache", "SqliteCache", "SystemClock", "compute_key"]
