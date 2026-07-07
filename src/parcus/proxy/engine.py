"""The proxy orchestration engine (non-streaming path).

Pipeline per request: detect dialect → parse to canonical (or pass through) → compress mutable
spans → decide cache eligibility → serve an exact cache hit or forward upstream → store a
cacheable response. Every optimization step **fails open**: on any error the original request
is forwarded unmodified and the real response returned — the proxy never breaks a harness or
changes a result to save tokens.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from parcus.cache.key import compute_key
from parcus.cache.policy import CachePolicy
from parcus.cache.seen import PrefixSeenSet
from parcus.cache.similarity import SimilarityCache
from parcus.cache.strategy import cache_strategy
from parcus.memory.compaction import compact_by_summary, compact_with_memory
from parcus.memory.provider import MemoryProvider, SharedMemoryProvider
from parcus.memory.summary import ExtractiveSummarizer, Summarizer
from parcus.model import (
    CachedResponse,
    CacheModel,
    CanonicalRequest,
    CompressionStats,
    Dialect,
    Role,
)
from parcus.obs import MetricsSink, NullSink, SavingsEvent, StageStat
from parcus.ports import CachePort, CompressorPort, MemoryPort, RedactorPort, TokenizerPort
from parcus.proxy.dialects import detect, parse, serialize
from parcus.proxy.upstream import UpstreamPort, UpstreamRequest, UpstreamResponse
from parcus.proxy.usage import parse_usage
from parcus.quota import RateLimiter
from parcus.tenant import derive_tenant, is_authorized
from parcus.tokenize import default_tokenizer

__all__ = ["EngineConfig", "ProxyEngine", "ProxyResult", "StreamPlan"]

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
        cache_inject: Whether to inject a provider prompt-cache breakpoint on a large stable
            prefix (M1b). Off by default; only explicit-breakpoint providers (Anthropic) act on
            it, and only when the prefix meets the provider's ``min_prefix_tokens``.
        cache_inject_repeat_aware: When injecting, require the prefix to have been **seen before**
            (within the provider cache window) so the ~1.25x cache-write premium is only paid when
            a repeat — and thus a ~0.1x cache-read — is likely (never-cost-more; issue #56). Default
            on; set off for unconditional "always inject" on the first sighting.
        parse_structured: When on, canonicalize **structured** requests (tool_use/tool_result/image
            blocks, OpenAI tool calls) by carrying those messages verbatim, instead of passing them
            through (M1d slice 1). Structured messages round-trip byte-for-byte and are left
            untouched by compression/injection/compaction here; the win in this slice is compact
            serialization applying to tool-using traffic. Off by default.
    """

    anthropic_upstream: str
    openai_upstream: str
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86_400
    salt: str = ""
    cache_inject: bool = False
    cache_inject_repeat_aware: bool = True
    parse_structured: bool = False
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


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """A prepared streaming request: either an early buffered response, or forward instructions.

    ``early`` is a non-``None`` :class:`ProxyResult` for a request that must not be forwarded
    (unroutable 502, unauthorized 401, rate-limited 429) — return it as-is. Otherwise forward
    ``out_body`` (the **compressed** request) to ``url`` with ``forward_headers`` and stream the
    response back untouched; ``meta`` carries the ``x-parcus-*`` header values.
    """

    early: ProxyResult | None
    url: str
    out_body: bytes
    forward_headers: tuple[tuple[str, str], ...]
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
        prefix_seen: PrefixSeenSet | None = None,
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
        self._prefix_seen = prefix_seen or PrefixSeenSet()

    @staticmethod
    def _upstream_url(base: str, path: str, query: str) -> str:
        """Join the routed base URL with the request path, preserving any query string.

        The query is forwarded verbatim (e.g. Gemini's ``?alt=sse``, which selects SSE framing —
        dropping it would break the stream). The host is fixed by :meth:`route`, so appending the
        client-supplied path + query adds no SSRF surface beyond the path forwarding that already
        happens (redirects are never followed).
        """
        url = base.rstrip("/") + path
        return f"{url}?{query}" if query else url

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

    def prepare_stream(
        self,
        dialect: Dialect,
        path: str,
        headers: list[tuple[str, str]],
        body: bytes,
        *,
        query: str = "",
    ) -> StreamPlan:
        """Authorize, rate-limit, and **compress** a streaming request for forwarding.

        Mirrors the front-half of :meth:`_handle` — routing, server-side tenant derivation + edge
        authorization, rate limiting, then canonicalize → memory → compress — but **never consults
        the response cache** (streaming responses are not cached). The caller forwards the returned
        compressed ``out_body`` and streams the response back untouched, so a streaming request
        still gets request-payload compression while its SSE response passes through byte-for-byte.
        Closing the prior gap where streaming bypassed authorization and rate limiting entirely.
        Fails open: an uncanonicalizable body forwards unchanged.
        """
        meta: dict[str, Any] = {"dialect": dialect.value, "cache": "stream-bypass"}
        base = self.route(dialect, headers)
        if base is None:
            return StreamPlan(
                early=ProxyResult(
                    status_code=502,
                    headers=[("content-type", "application/json")],
                    content=b'{"error":"parcus: unable to route request to a provider"}',
                    meta={**meta, "routed": False},
                ),
                url="",
                out_body=body,
                forward_headers=(),
                meta=meta,
            )
        url = self._upstream_url(base, path, query)
        scoped = self._config.multi_tenant or bool(self._config.allowed_tenants)
        tenant = derive_tenant(headers, salt=self._config.salt) if scoped else ""
        if tenant:
            meta["tenant"] = tenant
        if not is_authorized(tenant, self._config.allowed_tenants):
            return StreamPlan(
                early=ProxyResult(
                    status_code=401,
                    headers=[("content-type", "application/json")],
                    content=b'{"error":"parcus: tenant not authorized"}',
                    meta={**meta, "auth": "denied"},
                ),
                url=url,
                out_body=body,
                forward_headers=(),
                meta=meta,
            )
        if self._rate_limiter is not None:
            decision = self._rate_limiter.check(tenant)
            if not decision.allowed:
                return StreamPlan(
                    early=ProxyResult(
                        status_code=429,
                        headers=[
                            ("content-type", "application/json"),
                            ("retry-after", str(math.ceil(decision.retry_after))),
                        ],
                        content=b'{"error":"parcus: rate limit exceeded"}',
                        meta={**meta, "rate": "limited"},
                    ),
                    url=url,
                    out_body=body,
                    forward_headers=(),
                    meta=meta,
                )
        out_body = body
        canonical = self._canonicalize(dialect, body)
        if canonical is not None:
            working, memory_action = self._apply_memory(canonical, tenant)
            out_body, compressed, comp_stats = self._compress_request(working, body, tenant)
            meta["tokens_before"] = self._safe_count(canonical.text, canonical.model)
            meta["tokens_after"] = self._safe_count(compressed.text, compressed.model)
            meta["memory"] = memory_action
            meta["inject"] = "on" if compressed.cache_breakpoint is not None else "off"
            compacted = working is not canonical
            meta["stages"] = self._stage_stats(canonical, working, compacted, comp_stats)
        return StreamPlan(
            early=None,
            url=url,
            out_body=out_body,
            forward_headers=self._forward_headers(headers),
            meta=meta,
        )

    async def handle(
        self,
        method: str,
        path: str,
        headers: list[tuple[str, str]],
        body: bytes,
        *,
        query: str = "",
    ) -> ProxyResult:
        """Process one request and emit a savings metric (best-effort) around the core handler."""
        start = time.monotonic()
        result = await self._handle(method, path, headers, body, query=query)
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
                upstream_usage=meta.get("upstream_usage"),
            )
        )
        return result

    async def _handle(
        self,
        method: str,
        path: str,
        headers: list[tuple[str, str]],
        body: bytes,
        *,
        query: str = "",
    ) -> ProxyResult:
        """Process one buffered request end-to-end and return the result."""
        dialect = detect(path)
        base = self.route(dialect, headers)
        if base is None:
            return ProxyResult(
                status_code=502,
                headers=[("content-type", "application/json")],
                content=b'{"error":"parcus: unable to route request to a provider"}',
                meta={"routed": False},
            )

        url = self._upstream_url(base, path, query)
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
                content=b'{"error":"parcus: tenant not authorized"}',
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
                    content=b'{"error":"parcus: rate limit exceeded"}',
                    meta={**meta, "rate": "limited"},
                )

        canonical = self._canonicalize(dialect, body)
        if canonical is not None:
            working, memory_action = self._apply_memory(canonical, tenant)
            out_body, compressed, comp_stats = self._compress_request(working, body, tenant)
            meta["tokens_before"] = self._safe_count(canonical.text, canonical.model)
            meta["tokens_after"] = self._safe_count(compressed.text, compressed.model)
            meta["memory"] = memory_action
            meta["inject"] = "on" if compressed.cache_breakpoint is not None else "off"
            compacted = working is not canonical
            meta["stages"] = self._stage_stats(canonical, working, compacted, comp_stats)
            # Compacted bodies depend on evolving memory state, so they are not cached.
            cache_key = None if compacted else self._maybe_cache_key(canonical, tenant)
            if cache_key is not None:
                hit = self._cache_get(cache_key, tenant)
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
                # Best-effort: a similarity-index error must never break the response path.
                with contextlib.suppress(Exception):
                    self._similarity.remember(
                        text=canonical.text, key=cache_key, model=canonical.model, tenant=tenant
                    )
        # Capture the provider's billed usage + prompt-cache counts from the forwarded response
        # (ground truth + cache-interaction signal). Read-only and fail-open: never blocks.
        if 200 <= response.status_code < 300:
            usage = parse_usage(dialect, response.content)
            if usage is not None:
                meta["upstream_usage"] = usage
        return ProxyResult(
            status_code=response.status_code,
            headers=self._response_headers(response.headers),
            content=response.content,
            meta=meta,
        )

    # -- internal helpers (each fails open) -------------------------------------------

    def _safe_count(self, text: str, model: str | None) -> int:
        """Count tokens for observability, degrading to 0 on any tokenizer error.

        Token counts feed metrics only; a misbehaving tokenizer must never break the request
        path (fail open — defense in depth against a contract-violating adapter).
        """
        try:
            return self._tokenizer.count(text, model)
        except Exception:
            return 0

    def _cache_get(self, key: str, tenant: str) -> CachedResponse | None:
        """Read from the cache, treating any error as a miss (the cache is a perf layer)."""
        try:
            return self._cache.get(key, tenant=tenant)
        except Exception:
            return None

    def _canonicalize(self, dialect: Dialect, body: bytes) -> CanonicalRequest | None:
        if dialect is Dialect.UNKNOWN or not body:
            return None
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        return parse(dialect, decoded, structured=self._config.parse_structured)

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
        if any(m.raw is not None for m in canonical.messages):
            # Compaction over structured (tool_use/tool_result) turns is unsafe until designed
            # (M1d) — it could break tool pairing. Ingest only; leave the request intact.
            return canonical, "ingest"
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
        self, working: CanonicalRequest, original: bytes, tenant: str = ""
    ) -> tuple[bytes, CanonicalRequest, tuple[CompressionStats, ...]]:
        """Compress + re-serialise ``working`` to outbound bytes (fail open to the original).

        **Never-cost-more guard:** compression is guaranteed to shrink the *text* but not the
        provider's *BPE token* count — on rare inputs, removing whitespace can re-merge into one
        extra token. If the compressed request would tokenize to *more* tokens than its input, the
        compression is discarded and ``working`` is forwarded unchanged, so parcus never bills more
        than not compressing. (Applies to both the buffered and streaming paths.)
        """
        try:
            decoded = json.loads(original)
            compressed, stats = self._compressor.compress(working)
            chosen: CanonicalRequest
            chosen_stats: tuple[CompressionStats, ...]
            if self._safe_count(compressed.text, compressed.model) > self._safe_count(
                working.text, working.model
            ):
                # Compression expanded the token count — discard it, forward `working` as-is.
                chosen, chosen_stats = working, ()
            else:
                chosen, chosen_stats = compressed, stats
            # Prompt-cache injection (M1b) — mark a breakpoint on a large stable prefix so the
            # provider serves it from its cache next turn. Applied to whichever request we forward;
            # a no-op unless enabled and the provider/prefix qualify. Injection changes only the
            # cache_breakpoint marker, not the text, so the token metrics above are unaffected.
            chosen = self._inject_cache_breakpoint(chosen, tenant)
            # Serialise the forwarded body with compact separators (M1e): JSON structural
            # whitespace is non-semantic, so this losslessly minifies the request envelope —
            # notably verbose/pretty-printed tool schemas — without touching any string value
            # (message prose is a JSON string, preserved byte-for-byte). Providers parse JSON
            # identically. Passthrough (uncanonicalizable) bodies are forwarded untouched.
            encoded = json.dumps(
                serialize(chosen, decoded), ensure_ascii=False, separators=(",", ":")
            )
            return encoded.encode("utf-8"), chosen, chosen_stats
        except Exception:
            # Fail open: forward the original body unchanged; report no token delta.
            return original, working, ()

    def _inject_cache_breakpoint(
        self, request: CanonicalRequest, tenant: str = ""
    ) -> CanonicalRequest:
        """Mark a provider cache breakpoint on a qualifying stable prefix, else return unchanged.

        Gated (fail open): only when :attr:`EngineConfig.cache_inject` is on, the dialect's
        provider caches at explicit breakpoints, and the cacheable prefix (``system`` + ``tools`` +
        the protected turns) meets the provider's ``min_prefix_tokens`` — measured with the engine's
        tokenizer, which the pure strategy deliberately doesn't have. Below the minimum a breakpoint
        would not cache anyway, so injecting only adds block-list overhead; skip it.

        When ``cache_inject_repeat_aware`` is set, additionally require the prefix to have been seen
        before (per tenant, within the provider cache window) so the ~1.25x cache-write premium is
        only paid when a ~0.1x read is likely — never-cost-more in expectation (issue #56).
        """
        if not self._config.cache_inject:
            return request
        try:
            strategy = cache_strategy(request.dialect)
            capability = strategy.capability
            if capability.model is not CacheModel.EXPLICIT_BREAKPOINT:
                return request
            if _has_cache_control(request):
                # The harness already manages prompt caching — preserve it, don't add breakpoints
                # (avoids fighting its cache / exceeding the 4-breakpoint cap). M1a preservation.
                return request
            boundary = strategy.cacheable_boundary(request)
            if boundary is None or boundary < 1:
                return request
            prefix = (request.system or "") + (request.tools_json or "")
            prefix += "".join(m.text for m in request.messages[:boundary])
            if self._safe_count(prefix, request.model) < capability.min_prefix_tokens:
                return request
            if self._config.cache_inject_repeat_aware:
                digest = hashlib.sha256(f"{self._config.salt}\x00{prefix}".encode()).hexdigest()
                if not self._prefix_seen.record_and_check(digest, tenant=tenant):
                    return request  # first sighting — don't pay the cache-write premium yet
            return strategy.annotate(request)
        except Exception:
            return request

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
                    tokens_before=self._safe_count(canonical.text, canonical.model),
                    tokens_after=self._safe_count(working.text, working.model),
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
        try:
            # The secret check is inside the try so a misbehaving redactor fails *closed* for
            # caching (no key -> not cached) while the request still forwards (fails open).
            if not self._policy.should_cache(canonical, has_secret=self._redactor.has_secret):
                return None
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
        try:
            neighbour = self._similarity.lookup(
                text=canonical.text, model=canonical.model, tenant=tenant
            )
        except Exception:
            return None  # a misbehaving similarity index degrades to "no near-duplicate"
        if neighbour is None:
            return None
        return self._cache_get(neighbour, tenant)

    def _store(self, key: str, response: UpstreamResponse, tenant: str = "") -> None:
        # Best-effort: a cache write error must not affect the response already obtained.
        with contextlib.suppress(Exception):
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


def _has_cache_control(request: CanonicalRequest) -> bool:
    """Return whether the request already carries a ``cache_control`` breakpoint.

    Only structured (``raw``) messages can — a plain-text message's content is a string. Used to
    leave a harness that manages its own prompt caching untouched (don't inject a competing marker).
    """
    for message in request.messages:
        if message.raw is None:
            continue
        content = message.raw.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


def _content_type(headers: tuple[tuple[str, str], ...]) -> str | None:
    for key, value in headers:
        if key.lower() == "content-type":
            return value
    return None


def _request_id(headers: list[tuple[str, str]]) -> str:
    """Return an inbound correlation id (``x-request-id``/``x-correlation-id``) or a fresh one."""
    lower = {k.lower(): v for k, v in headers}
    return lower.get("x-request-id") or lower.get("x-correlation-id") or uuid.uuid4().hex
