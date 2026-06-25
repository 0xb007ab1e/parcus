"""The decision of *whether* a request may be cached (distinct from the store itself).

Caching is skipped when disabled, when the request text matches a configured no-cache
pattern (e.g. an auth flow), or — as defense in depth — when it contains a detected
credential. Detecting a credential only suppresses caching; it never changes the request that
is forwarded upstream.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from parcus.model import CanonicalRequest

__all__ = ["CachePolicy"]


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Configuration governing which requests are eligible for caching.

    Args:
        enabled: Master switch; when False nothing is cached.
        ttl_seconds: Time-to-live applied to stored entries.
        nocache_patterns: Compiled regexes; a request whose text matches any is never cached.
        bypass_on_secret: When True, requests containing a detected credential are not cached.
    """

    enabled: bool = True
    ttl_seconds: int = 86_400
    nocache_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    bypass_on_secret: bool = True

    @classmethod
    def from_patterns(
        cls,
        patterns: Iterable[str] = (),
        *,
        enabled: bool = True,
        ttl_seconds: int = 86_400,
        bypass_on_secret: bool = True,
    ) -> CachePolicy:
        """Build a policy, compiling ``patterns`` (regex strings) to no-cache matchers."""
        compiled = tuple(re.compile(p) for p in patterns)
        return cls(
            enabled=enabled,
            ttl_seconds=ttl_seconds,
            nocache_patterns=compiled,
            bypass_on_secret=bypass_on_secret,
        )

    def should_cache(
        self,
        request: CanonicalRequest,
        *,
        has_secret: Callable[[str], bool] | None = None,
    ) -> bool:
        """Return whether ``request`` may be cached under this policy.

        Args:
            request: The request being considered.
            has_secret: Optional credential detector (e.g. ``Redactor.has_secret``); used only
                when ``bypass_on_secret`` is set.

        Returns:
            True if the request is eligible for caching, else False.
        """
        if not self.enabled:
            return False
        text = request.text
        if any(pattern.search(text) for pattern in self.nocache_patterns):
            return False
        if self.bypass_on_secret and has_secret is not None and has_secret(text):
            return False
        return True
