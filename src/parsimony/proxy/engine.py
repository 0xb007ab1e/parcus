"""The proxy orchestration engine (non-streaming path).

Pipeline per request: detect dialect → parse to canonical (or pass through) → compress mutable
spans → decide cache eligibility → serve an exact cache hit or forward upstream → store a
cacheable response. Every optimization step **fails open**: on any error the original request
is forwarded unmodified and the real response returned — the proxy never breaks a harness or
changes a result to save tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from parsimony.cache.key import compute_key
from parsimony.cache.policy import CachePolicy
from parsimony.model import CachedResponse, CanonicalRequest, Dialect
from parsimony.ports import CachePort, CompressorPort, RedactorPort
from parsimony.proxy.dialects import detect, parse, serialize
from parsimony.proxy.upstream import UpstreamPort, UpstreamRequest, UpstreamResponse

__all__ = ["EngineConfig", "ProxyEngine", "ProxyResult"]

# Request headers we must not forward verbatim (recomputed by the client or unsafe to relay).
# accept-encoding is dropped so responses come back identity-encoded and replay cleanly.
_DROP_REQUEST_HEADERS = {"host", "content-length", "accept-encoding", "connection"}
# Response headers we must not relay back (the ASGI server sets framing itself).
_DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Engine configuration resolved from settings at the composition root.

    Args:
        anthropic_upstream: Base URL for Anthropic.
        openai_upstream: Base URL for OpenAI.
        cache_enabled: Whether the response cache is active.
        cache_ttl_seconds: TTL applied to stored responses.
        salt: Per-install salt for cache-key domain separation.
    """

    anthropic_upstream: str
    openai_upstream: str
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86_400
    salt: str = ""


@dataclass(frozen=True, slots=True)
class ProxyResult:
    """The buffered result the app turns into an HTTP response."""

    status_code: int
    headers: list[tuple[str, str]]
    content: bytes
    meta: dict[str, Any] = field(default_factory=dict)


class ProxyEngine:
    """Wires compression and caching around upstream forwarding (non-streaming)."""

    def __init__(
        self,
        *,
        upstream: UpstreamPort,
        compressor: CompressorPort,
        cache: CachePort,
        redactor: RedactorPort,
        policy: CachePolicy,
        config: EngineConfig,
    ) -> None:
        """Inject the upstream adapter, optimization components, and resolved config."""
        self._upstream = upstream
        self._compressor = compressor
        self._cache = cache
        self._redactor = redactor
        self._policy = policy
        self._config = config

    def route(self, dialect: Dialect, headers: list[tuple[str, str]]) -> str | None:
        """Return the upstream base URL for a request, or ``None`` if it cannot be routed."""
        if dialect is Dialect.ANTHROPIC:
            return self._config.anthropic_upstream
        if dialect is Dialect.OPENAI:
            return self._config.openai_upstream
        lower = {k.lower(): v for k, v in headers}
        if "x-api-key" in lower or "anthropic-version" in lower:
            return self._config.anthropic_upstream
        if "authorization" in lower:
            return self._config.openai_upstream
        return None

    async def handle(
        self, method: str, path: str, headers: list[tuple[str, str]], body: bytes
    ) -> ProxyResult:
        """Process one buffered request end-to-end and return the result."""
        dialect = detect(path)
        base = self.route(dialect, headers)
        if base is None:
            return ProxyResult(
                status_code=502,
                headers=[("content-type", "application/json")],
                content=b'{"error":"parsimony: unable to route request to a provider"}',
                meta={"routed": False},
            )

        url = base.rstrip("/") + path
        out_body = body
        meta: dict[str, Any] = {"dialect": dialect.value, "cache": "off"}
        cache_key: str | None = None

        canonical = self._canonicalize(dialect, body)
        if canonical is not None:
            out_body, stats = self._compress(canonical, body)
            if stats is not None:
                meta["tokens_before"] = stats[0]
                meta["tokens_after"] = stats[1]
            cache_key = self._maybe_cache_key(canonical)
            if cache_key is not None:
                hit = self._cache.get(cache_key)
                if hit is not None:
                    return self._from_cache(hit, meta)
                meta["cache"] = "miss"

        response = await self._upstream.send(
            UpstreamRequest(
                method=method,
                url=url,
                headers=self._forward_headers(headers),
                content=out_body,
            )
        )
        if cache_key is not None and 200 <= response.status_code < 300:
            self._store(cache_key, response)
        return ProxyResult(
            status_code=response.status_code,
            headers=self._response_headers(response.headers),
            content=response.content,
            meta=meta,
        )

    # -- internal helpers (each fails open) -------------------------------------------

    def _canonicalize(self, dialect: Dialect, body: bytes) -> CanonicalRequest | None:
        if dialect is Dialect.UNKNOWN or not body:
            return None
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        return parse(dialect, decoded)

    def _compress(
        self, canonical: CanonicalRequest, original: bytes
    ) -> tuple[bytes, tuple[int, int] | None]:
        try:
            compressed, stats = self._compressor.compress(canonical)
            decoded = json.loads(original)
            new_body = serialize(compressed, decoded)
            encoded = json.dumps(new_body, ensure_ascii=False).encode("utf-8")
            tokens = (stats[0].tokens_before, stats[0].tokens_after) if stats else None
            return encoded, tokens
        except Exception:
            # Fail open: forward the original body unchanged.
            return original, None

    def _maybe_cache_key(self, canonical: CanonicalRequest) -> str | None:
        if not self._config.cache_enabled or canonical.stream:
            return None
        if not self._policy.should_cache(canonical, has_secret=self._redactor.has_secret):
            return None
        try:
            return compute_key(canonical, salt=self._config.salt)
        except Exception:
            return None

    def _store(self, key: str, response: UpstreamResponse) -> None:
        self._cache.put(
            key,
            CachedResponse(
                status_code=response.status_code,
                body=response.content,
                content_type=_content_type(response.headers),
            ),
            self._config.cache_ttl_seconds,
        )

    def _from_cache(self, hit: CachedResponse, meta: dict[str, Any]) -> ProxyResult:
        headers = [("content-type", hit.content_type or "application/json")]
        return ProxyResult(
            status_code=hit.status_code,
            headers=headers,
            content=hit.body,
            meta={**meta, "cache": "hit"},
        )

    @staticmethod
    def _forward_headers(headers: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple((k, v) for k, v in headers if k.lower() not in _DROP_REQUEST_HEADERS)

    @staticmethod
    def _response_headers(headers: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
        return [(k, v) for k, v in headers if k.lower() not in _DROP_RESPONSE_HEADERS]


def _content_type(headers: tuple[tuple[str, str], ...]) -> str | None:
    for key, value in headers:
        if key.lower() == "content-type":
            return value
    return None
