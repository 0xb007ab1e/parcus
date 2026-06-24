"""The per-request savings event.

Carries only **counts and metadata** — never prompt or response content — so it is PII-safe by
construction (redaction-by-omission; see master §5 and ``topic-logging-observability``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["SavingsEvent"]


@dataclass(frozen=True, slots=True)
class SavingsEvent:
    """A single request's token-savings outcome.

    Args:
        request_id: Correlation id (from an inbound header or generated).
        dialect: Detected provider dialect (``anthropic`` / ``openai`` / ``unknown``).
        cache: Cache outcome (``hit`` / ``miss`` / ``off``).
        canonicalized: Whether the request was canonicalised (eligible for compression).
        tokens_before: Input tokens before compression (0 when not canonicalised).
        tokens_after: Input tokens after compression (0 when not canonicalised).
        status_code: HTTP status returned to the client.
        duration_ms: Proxy-side handling time in milliseconds.
    """

    request_id: str
    dialect: str
    cache: str
    canonicalized: bool
    tokens_before: int
    tokens_after: int
    status_code: int
    duration_ms: float

    @property
    def tokens_saved(self) -> int:
        """Tokens removed for this request (never negative)."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def ratio(self) -> float:
        """Fraction of input tokens removed, in ``[0.0, 1.0]`` (0 when no input)."""
        if self.tokens_before <= 0:
            return 0.0
        return self.tokens_saved / self.tokens_before

    def to_dict(self) -> dict[str, Any]:
        """Render as a flat, stable-schema dict for structured logging (no content)."""
        return {
            "event": "savings",
            "request_id": self.request_id,
            "dialect": self.dialect,
            "cache": self.cache,
            "canonicalized": self.canonicalized,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "ratio": round(self.ratio, 4),
            "status_code": self.status_code,
            "duration_ms": round(self.duration_ms, 2),
        }
