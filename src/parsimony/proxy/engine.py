"""The proxy orchestration engine (non-streaming path).

Pipeline per request: detect dialect → parse to canonical (or pass through) → compress mutable
spans → decide cache eligibility → serve an exact cache hit or forward upstream → store a
cacheable response. Every optimization step **fails open**: on any error the original request
is forwarded unmodified and the real response returned — the proxy never breaks a harness or
changes a result to save tokens.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from parsimony.cache.key import compute_key
from parsimony.cache.policy import CachePolicy
from parsimony.cache.similarity import SimilarityCache
from parsimony.memory.compaction import compact_by_summary, compact_with_memory
from parsimony.memory.provider import MemoryProvider, SharedMemoryProvider
from parsimony.memory.summary import ExtractiveSummarizer, Summarizer
from parsimony.model import CachedResponse, CanonicalRequest, CompressionStats, Dialect, Role
from parsimony.obs import MetricsSink, NullSink, SavingsEvent, StageStat
from parsimony.ports import CachePort, CompressorPort, MemoryPort, RedactorPort, TokenizerPort
from parsimony.proxy.dialects import detect, parse, serialize
from parsimony.proxy.upstream import UpstreamPort, UpstreamRequest, UpstreamResponse
from parsimony.quota import RateLimiter
from parsimony.tenant import derive_tenant, is_authorized
from parsimony.tokenize import default_tokenizer

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
    # Graph memory (off by default). ingest builds the graph; inject (Track B) compacts via
    # retrieval; summarize (Track C) replaces older turns with a rolling summary.
    memory_enabled: bool = False
    memory_inject: bool = False
    memory_summarize: bool = False
    memory_keep_recent: int = 4
    memory_retrieve: int = 3
    memory_summary_items: int = 5
    memory_min_messages: int = 8
    # Hosted/multi-tenant mode (off by default). When on, the tenant is derived server-side from
    # the inbound credential and folded into the cache key so tenants never share cached data.
    multi_tenant: bool = False
    # Optional edge authorization (hosted mode). Empty = open (provider still authenticates the
    # forwarded credential). Non-empty = fail-closed allow-list of permitted tenant ids.
    allowed_tenants: frozenset[str] = frozenset()


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
        metrics: MetricsSink | None = None,
        memory: MemoryPort | None = None,
        memory_provider: MemoryProvider | None = None,
        rate_limiter: RateLimiter | None = None,
        similarity: SimilarityCache | None = None,
        tokenizer: TokenizerPort | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        """Inject the upstream adapter, optimization components, config, metrics, and memory.

        ``memory_provider`` resolves the per-tenant memory; if omitted, ``memory`` is wrapped in a
        :class:`SharedMemoryProvider` (one graph for all tenants — single-tenant behaviour).
        ``rate_limiter`` (when given) caps per-tenant request rate; ``None`` disables limiting.
        """
        self._upstream = upstream
        self._compressor = compressor
        self._cache = cache
        self._redactor = redactor
        self._policy = policy
        self._config = config
        self._metrics = metrics or NullSink()
        self._memory_provider = memory_provider or SharedMemoryProvider(memory)
        self._rate_limiter = rate_limiter
        self._similarity = similarity
        self._tokenizer = tokenizer or default_tokenizer()
        self._summarizer = summarizer or ExtractiveSummarizer()

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
        """Process one request and emit a savings metric (best-effort) around the core handler."""
        start = time.monotonic()
        result = await self._handle(method, path, headers, body)
        meta = result.meta
        self._metrics.record(
            SavingsEvent(
                request_id=_request_id(headers),
                dialect=str(meta.get("dialect", "unknown")),
                cache=str(meta.get("cache", "off")),
                canonicalized="tokens_before" in meta,
                tokens_before=int(meta.get("tokens_before", 0)),
                tokens_after=int(meta.get("tokens_after", 0)),
                status_code=result.status_code,
                duration_ms=(time.monotonic() - start) * 1000.0,
                stages=tuple(meta.get("stages", ())),
                tenant=str(meta.get("tenant", "")),
            )
        )
        return result

    async def _handle(
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
        # Tenant is derived from the credential server-side (empty in single-tenant mode). It is
        # also derived when an edge allow-list is configured, so authorization can be enforced.
        scoped = self._config.multi_tenant or bool(self._config.allowed_tenants)
        tenant = derive_tenant(headers, salt=self._config.salt) if scoped else ""
        if tenant:
            meta["tenant"] = tenant  # content-free attribution; not exposed as a response header
        if not is_authorized(tenant, self._config.allowed_tenants):
            # Fail closed: an unlisted/anonymous tenant never reaches an upstream.
            return ProxyResult(
                status_code=401,
                headers=[("content-type", "application/json")],
                content=b'{"error":"parsimony: tenant not authorized"}',
                meta={**meta, "auth": "denied"},
            )
        if self._rate_limiter is not None:
            decision = self._rate_limiter.check(tenant)
            if not decision.allowed:
                # Fail closed against abuse: shed the request before it reaches an upstream.
                return ProxyResult(
                    status_code=429,
                    headers=[
                        ("content-type", "application/json"),
                        ("retry-after", str(math.ceil(decision.retry_after))),
                    ],
                    content=b'{"error":"parsimony: rate limit exceeded"}',
                    meta={**meta, "rate": "limited"},
                )

        canonical = self._canonicalize(dialect, body)
        if canonical is not None:
            working, memory_action = self._apply_memory(canonical, tenant)
            out_body, compressed, comp_stats = self._compress_request(working, body)
            before = self._tokenizer.count(canonical.text, canonical.model)
            meta["tokens_before"] = before
            meta["tokens_after"] = self._tokenizer.count(compressed.text, compressed.model)
            meta["memory"] = memory_action
            compacted = working is not canonical
            meta["stages"] = self._stage_stats(canonical, working, compacted, comp_stats)
            # Compacted bodies depend on evolving memory state, so they are not cached.
            cache_key = None if compacted else self._maybe_cache_key(canonical, tenant)
            if cache_key is not None:
                hit = self._cache.get(cache_key, tenant=tenant)
                if hit is not None:
                    return self._from_cache(hit, meta)
                meta["cache"] = "miss"
                similar = self._similar_hit(canonical, tenant)
                if similar is not None:
                    return self._from_cache(similar, meta, outcome="similar")

        response = await self._upstream.send(
            UpstreamRequest(
                method=method,
                url=url,
                headers=self._forward_headers(headers),
                content=out_body,
            )
        )
        if cache_key is not None and 200 <= response.status_code < 300:
            self._store(cache_key, response, tenant)
            if self._similarity is not None and canonical is not None:
                self._similarity.remember(
                    text=canonical.text, key=cache_key, model=canonical.model, tenant=tenant
                )
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

    def _apply_memory(
        self, canonical: CanonicalRequest, tenant: str
    ) -> tuple[CanonicalRequest, str]:
        """Ingest into the tenant's memory and optionally compact the request (fail open).

        The memory is resolved per tenant, so one tenant's context can never be ingested into or
        retrieved from another's graph. Returns the (possibly compacted) request and an action
        label (``off``/``ingest``/``compact``/``summary``) for observability.
        """
        if not self._config.memory_enabled:
            return canonical, "off"
        memory = self._memory_provider.for_tenant(tenant)
        if memory is None:
            return canonical, "off"
        try:
            memory.ingest(canonical)
        except Exception:
            return canonical, "off"
        if self._config.memory_summarize:
            try:
                summarized = compact_by_summary(
                    canonical,
                    self._summarizer,
                    keep_recent=self._config.memory_keep_recent,
                    max_items=self._config.memory_summary_items,
                    min_messages=self._config.memory_min_messages,
                )
            except Exception:
                return canonical, "ingest"
            return summarized, "summary" if summarized is not canonical else "ingest"
        if not self._config.memory_inject:
            return canonical, "ingest"
        try:
            compacted = compact_with_memory(
                canonical,
                memory,
                keep_recent=self._config.memory_keep_recent,
                retrieve=self._config.memory_retrieve,
                min_messages=self._config.memory_min_messages,
            )
        except Exception:
            return canonical, "ingest"
        return compacted, "compact" if compacted is not canonical else "ingest"

    def _compress_request(
        self, working: CanonicalRequest, original: bytes
    ) -> tuple[bytes, CanonicalRequest, tuple[CompressionStats, ...]]:
        """Compress + re-serialise ``working`` to outbound bytes (fail open to the original)."""
        try:
            compressed, stats = self._compressor.compress(working)
            decoded = json.loads(original)
            encoded = json.dumps(serialize(compressed, decoded), ensure_ascii=False).encode("utf-8")
            return encoded, compressed, stats
        except Exception:
            # Fail open: forward the original body unchanged; report no token delta.
            return original, working, ()

    def _stage_stats(
        self,
        canonical: CanonicalRequest,
        working: CanonicalRequest,
        compacted: bool,
        comp_stats: tuple[CompressionStats, ...],
    ) -> list[StageStat]:
        """Build the per-stage reduction + accuracy breakdown for observability."""
        stages: list[StageStat] = []
        if compacted:
            stages.append(
                StageStat(
                    stage="memory",
                    tokens_before=self._tokenizer.count(canonical.text, canonical.model),
                    tokens_after=self._tokenizer.count(working.text, working.model),
                    ok=self._memory_structural_ok(working),
                )
            )
        stages.extend(
            StageStat(
                stage=stat.step,
                tokens_before=stat.tokens_before,
                tokens_after=stat.tokens_after,
                ok=stat.ok,
            )
            for stat in comp_stats
        )
        return stages

    @staticmethod
    def _memory_structural_ok(request: CanonicalRequest) -> bool:
        """A model-free structural check for a memory-compacted request (Anthropic-valid shape)."""
        return bool(request.messages) and request.messages[0].role is Role.USER

    def _maybe_cache_key(self, canonical: CanonicalRequest, tenant: str = "") -> str | None:
        if not self._config.cache_enabled or canonical.stream:
            return None
        if not self._policy.should_cache(canonical, has_secret=self._redactor.has_secret):
            return None
        try:
            # Namespace the key per tenant so one tenant can never read another's cached
            # response (BOLA). Empty tenant (single-tenant mode) leaves the salt unchanged.
            salt = f"{self._config.salt}|t:{tenant}" if tenant else self._config.salt
            return compute_key(canonical, salt=salt)
        except Exception:
            return None

    def _similar_hit(self, canonical: CanonicalRequest, tenant: str) -> CachedResponse | None:
        """Return a near-duplicate's cached response (same model + tenant), or ``None``.

        Consults the similarity index (when enabled) on an exact miss, then fetches the
        neighbour's response from the exact cache so TTL/eviction still apply. Fails open.
        """
        if self._similarity is None or canonical.stream:
            return None
        neighbour = self._similarity.lookup(
            text=canonical.text, model=canonical.model, tenant=tenant
        )
        if neighbour is None:
            return None
        return self._cache.get(neighbour, tenant=tenant)

    def _store(self, key: str, response: UpstreamResponse, tenant: str = "") -> None:
        self._cache.put(
            key,
            CachedResponse(
                status_code=response.status_code,
                body=response.content,
                content_type=_content_type(response.headers),
            ),
            self._config.cache_ttl_seconds,
            tenant=tenant,
        )

    def _from_cache(
        self, hit: CachedResponse, meta: dict[str, Any], *, outcome: str = "hit"
    ) -> ProxyResult:
        headers = [("content-type", hit.content_type or "application/json")]
        return ProxyResult(
            status_code=hit.status_code,
            headers=headers,
            content=hit.body,
            meta={**meta, "cache": outcome},
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


def _request_id(headers: list[tuple[str, str]]) -> str:
    """Return an inbound correlation id (``x-request-id``/``x-correlation-id``) or a fresh one."""
    lower = {k.lower(): v for k, v in headers}
    return lower.get("x-request-id") or lower.get("x-correlation-id") or uuid.uuid4().hex
