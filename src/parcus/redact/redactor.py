"""Regex-based redactor: mask secrets/PII and detect credentials for the cache bypass."""

from __future__ import annotations

from parcus.model import RedactionReport
from parcus.redact.patterns import DEFAULT_PATTERNS, SecretPattern

__all__ = ["Redactor", "placeholder_for"]


def placeholder_for(name: str) -> str:
    """Return the masking placeholder used for a detection category ``name``."""
    return f"«REDACTED:{name}»"


class Redactor:
    """Mask sensitive spans before text is persisted or logged.

    Implements :class:`parcus.ports.RedactorPort`. Model-free and deterministic; it is
    never applied to the request forwarded upstream nor to a replayed cache response.

    Args:
        patterns: Detection rules to apply. Defaults to
            :data:`parcus.redact.patterns.DEFAULT_PATTERNS`.
    """

    def __init__(self, patterns: tuple[SecretPattern, ...] | None = None) -> None:
        """Initialise the redactor with an optional custom pattern set."""
        self._patterns = patterns if patterns is not None else DEFAULT_PATTERNS

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        """Mask every matching span and report the categories and count found.

        Args:
            text: The text to scrub.

        Returns:
            A tuple of the masked text and a :class:`RedactionReport`.
        """
        total = 0
        found: set[str] = set()
        out = text
        for pattern in self._patterns:
            out, count = pattern.regex.subn(placeholder_for(pattern.name), out)
            if count:
                total += count
                found.add(pattern.name)
        return out, RedactionReport(total=total, categories=tuple(sorted(found)))

    def has_secret(self, text: str) -> bool:
        """Return whether ``text`` contains a credential-class secret (not mere PII)."""
        return any(p.is_secret and p.regex.search(text) is not None for p in self._patterns)
