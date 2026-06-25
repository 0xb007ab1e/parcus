"""The per-request savings event.

Carries only **counts and metadata** — never prompt or response content — so it is PII-safe by
construction (redaction-by-omission; see master §5 and ``topic-logging-observability``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["SavingsEvent", "StageStat"]


@dataclass(frozen=True, slots=True)
class StageStat:
    """Reduction + accuracy for one pipeline stage of a request.

    Args:
        stage: Stage name (e.g. ``memory``, ``lossless``, ``filler``).
        tokens_before: Tokens entering the stage.
        tokens_after: Tokens leaving the stage.
        ok: The stage's model-free invariant self-check (held = True), or ``None`` if the
            stage has no runtime invariant (its accuracy comes from the offline eval gate).
    """

    stage: str
    tokens_before: int
    tokens_after: int
    ok: bool | None = None

    @property
    def tokens_saved(self) -> int:
        """Tokens removed by this stage (never negative)."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def ratio(self) -> float:
        """Fraction of stage-input tokens removed, in ``[0.0, 1.0]`` (0 when no input)."""
        if self.tokens_before <= 0:
            return 0.0
        return self.tokens_saved / self.tokens_before

    def to_dict(self) -> dict[str, Any]:
        """Render as a flat dict (counts only)."""
        return {
            "stage": self.stage,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "ok": self.ok,
        }


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
        stages: Per-stage reduction + accuracy breakdown (memory, lossless, filler, …).
        tenant: Opaque, content-free tenant id for per-tenant attribution (empty in
            single-tenant mode). Never the raw credential — see :mod:`parcus.tenant`.
    """

    request_id: str
    dialect: str
    cache: str
    canonicalized: bool
    tokens_before: int
    tokens_after: int
    status_code: int
    duration_ms: float
    stages: tuple[StageStat, ...] = field(default_factory=tuple)
    tenant: str = ""

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
            "tenant": self.tenant,
            "stages": [s.to_dict() for s in self.stages],
        }
