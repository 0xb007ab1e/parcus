"""Tests for the proxy engine: compression, caching, routing, and fail-open behaviour."""

from __future__ import annotations

import json

from parcus.cache import CachePolicy, SqliteCache
from parcus.compress import LosslessCompressor
from parcus.model import CanonicalRequest, CompressionStats, Message
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor

OK = UpstreamResponse(200, (("content-type", "application/json"),), b'{"r":1}')


class FakeUpstream:
    """Records the forwarded request and returns a canned response."""

    def __init__(self, response: UpstreamResponse = OK) -> None:
        self._response = response
        self.calls = 0
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.calls += 1
        self.last = request
        return self._response


class BoomCompressor:
    """A compressor that always raises (to exercise the engine's fail-open path)."""

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        raise RuntimeError("boom")


class ExpandingCompressor:
    """A (perverse) compressor that makes the request longer — exercises the never-cost-more guard.

    Real compression only shrinks text, but BPE token count can rarely tick up; this fake forces
    that condition deterministically so we can assert the engine discards it.
    """

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        grown = tuple(
            Message(
                role=m.role,
                spans=tuple(
                    s.with_text(s.text + " extra padding words added to grow the token count")
                    if s.mutable
                    else s
                    for s in m.spans
                ),
            )
            for m in request.messages
        )
        expanded = CanonicalRequest(
            dialect=request.dialect,
            model=request.model,
            messages=grown,
            system=request.system,
            stream=request.stream,
            tools_json=request.tools_json,
        )
        return expanded, (CompressionStats(step="evil", tokens_before=1, tokens_after=99),)


class FixedTokenizer:
    """Returns a constant token count regardless of input (drives the min-prefix gate in tests)."""

    def __init__(self, count: int) -> None:
        self._count = count

    def count(self, text: str, model: str | None = None) -> int:
        return self._count


def _engine(upstream: FakeUpstream, **kw: object) -> ProxyEngine:
    return ProxyEngine(
        upstream=upstream,
        compressor=kw.get("compressor", LosslessCompressor()),  # type: ignore[arg-type]
        cache=kw.get("cache", SqliteCache()),  # type: ignore[arg-type]
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream="https://a.test",
            openai_upstream="https://o.test",
            cache_enabled=bool(kw.get("cache_enabled", True)),
            cache_inject=bool(kw.get("cache_inject", False)),
            cache_inject_repeat_aware=bool(kw.get("cache_inject_repeat_aware", True)),
            multi_tenant=bool(kw.get("multi_tenant", False)),
            allowed_tenants=kw.get("allowed_tenants", frozenset()),  # type: ignore[arg-type]
        ),
        metrics=kw.get("metrics"),  # type: ignore[arg-type]
        rate_limiter=kw.get("rate_limiter"),  # type: ignore[arg-type]
        similarity=kw.get("similarity"),  # type: ignore[arg-type]
        tokenizer=kw.get("tokenizer"),  # type: ignore[arg-type]
    )


class SpySink:
    """Records savings events for assertions."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def record(self, event: object) -> None:
        self.events.append(event)


def _anthropic(content: str, *, system: str | None = None, stream: bool = False) -> bytes:
    body: dict[str, object] = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": content}],
    }
    if system is not None:
        body["system"] = system
    if stream:
        body["stream"] = True
    return json.dumps(body).encode()


def _anthropic_multi(*contents: str, system: str | None = None) -> bytes:
    """An Anthropic body with multiple user turns (for exercising cache-breakpoint placement)."""
    body: dict[str, object] = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": c} for c in contents],
    }
    if system is not None:
        body["system"] = system
    return json.dumps(body).encode()


class TestCompactSerialization:
    """M1e: the forwarded canonicalized body is serialised as compact (lossless) JSON."""

    async def test_body_uses_compact_separators(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("hi", system="s"))
        sent = up.last.content
        assert b'"messages":[' in sent  # compact: no space after ':' or before '['
        assert b'"content":"hi"' in sent
        assert b'"system":"s"' in sent
        assert b'": "' not in sent  # no pretty ': ' structural spacing (safe for these inputs)

    async def test_minifies_tool_schema_and_preserves_meaning(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
        body = json.dumps(  # verbose, whitespace-heavy (as a harness might pretty-print)
            {"model": "claude-x", "messages": [{"role": "user", "content": "go"}], "tools": tools},
            indent=2,
        ).encode()
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        sent = up.last.content
        assert b"\n" not in sent  # structural newlines from indent=2 are gone
        assert len(sent) < len(body)  # strictly smaller
        parsed = json.loads(sent)
        assert parsed["tools"] == tools  # meaning preserved exactly
        assert parsed["messages"][0]["content"] == "go"


class TestCacheInjection:
    """M1b: gated provider prompt-cache breakpoint injection on the forwarded request."""

    async def test_off_by_default(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)  # cache_inject defaults False
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_multi("a", "b"))
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "a"  # plain string, no cache_control

    async def test_always_mode_injects_first_request(self) -> None:
        # repeat_aware off → inject on the first sighting (the M1b-2 "always" behaviour).
        up = FakeUpstream()
        eng = _engine(
            up,
            cache_enabled=False,
            cache_inject=True,
            cache_inject_repeat_aware=False,
            tokenizer=FixedTokenizer(5000),
        )
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_multi("a", "b"))
        sent = json.loads(up.last.content)
        # boundary is 1 (protect the first turn) → cache_control on messages[0].
        assert sent["messages"][0]["content"] == [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}
        ]
        assert sent["messages"][1]["content"] == "b"  # volatile tail stays plain

    async def test_repeat_aware_skips_first_then_injects_second(self) -> None:
        # Default (repeat_aware on): the first sighting isn't injected (no write premium yet);
        # the second identical request is (a read is now likely). Same engine → shared seen-set.
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False, cache_inject=True, tokenizer=FixedTokenizer(5000))
        body = _anthropic_multi("a", "b")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        first = json.loads(up.last.content)
        assert first["messages"][0]["content"] == "a"  # first sighting → not injected
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        second = json.loads(up.last.content)
        assert second["messages"][0]["content"] == [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}
        ]

    async def test_skips_when_prefix_below_min_tokens(self) -> None:
        up = FakeUpstream()
        eng = _engine(
            up,
            cache_enabled=False,
            cache_inject=True,
            cache_inject_repeat_aware=False,
            tokenizer=FixedTokenizer(10),
        )
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_multi("a", "b"))
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "a"  # below 4096 → no breakpoint

    async def test_skips_single_message_no_protectable_prefix(self) -> None:
        up = FakeUpstream()
        eng = _engine(
            up,
            cache_enabled=False,
            cache_inject=True,
            cache_inject_repeat_aware=False,
            tokenizer=FixedTokenizer(5000),
        )
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic_multi("only"))
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "only"  # boundary 0 → nothing to inject

    async def test_skips_for_openai_no_explicit_breakpoint(self) -> None:
        up = FakeUpstream()
        eng = _engine(
            up,
            cache_enabled=False,
            cache_inject=True,
            cache_inject_repeat_aware=False,
            tokenizer=FixedTokenizer(5000),
        )
        body = json.dumps(
            {
                "model": "gpt-x",
                "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
            }
        ).encode()
        await eng.handle("POST", "/v1/chat/completions", [("authorization", "Bearer k")], body)
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "a"  # OpenAI = automatic-prefix → no injection


class TestForwardingAndCompression:
    async def test_compresses_mutable_text_and_preserves_headers(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        body = _anthropic("hello   \n\n\n\n\nworld   ", system="sys   \n\n\n\nmore")
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "hello\n\nworld"
        assert sent["system"] == "sys\n\nmore"
        assert result.meta["tokens_before"] >= result.meta["tokens_after"]
        assert ("x-api-key", "k") in up.last.headers

    async def test_passthrough_when_not_canonicalizable(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        body = json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            }
        ).encode()
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.last.content == body  # forwarded unchanged
        assert result.meta["cache"] == "off"

    async def test_fails_open_when_compressor_raises(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, compressor=BoomCompressor(), cache_enabled=False)
        body = _anthropic("data   \n\n\n\n")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.last.content == body  # original forwarded despite the failure

    async def test_never_cost_more_guard_discards_token_expanding_compression(self) -> None:
        # If compression would produce MORE tokens than the input, forward the original unchanged.
        up = FakeUpstream()
        eng = _engine(up, compressor=ExpandingCompressor(), cache_enabled=False)
        body = _anthropic("keep me exactly as sent")
        result = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "keep me exactly as sent"  # not expanded
        # meta reflects no expansion (compression discarded)
        assert result.meta["tokens_after"] <= result.meta["tokens_before"]


class TestCaching:
    async def test_miss_then_hit_skips_second_upstream_call(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("please compute")
        first = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert first.meta["cache"] == "miss"
        assert second.meta["cache"] == "hit"
        assert up.calls == 1
        assert second.content == b'{"r":1}'

    async def test_credential_request_is_not_cached(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("token sk-ant-api03-" + "A" * 24)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.calls == 2  # never served from cache

    async def test_streaming_request_is_not_cached(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())
        body = _anthropic("hi", stream=True)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert up.calls == 2


class TestRouting:
    async def test_unknown_path_without_auth_returns_502(self) -> None:
        up = FakeUpstream()
        eng = _engine(up)
        result = await eng.handle("GET", "/v1/models", [], b"")
        assert result.status_code == 502
        assert up.calls == 0

    async def test_unknown_path_routed_by_authorization_header(self) -> None:
        up = FakeUpstream()
        eng = _engine(up)
        result = await eng.handle("GET", "/v1/models", [("authorization", "Bearer x")], b"")
        assert result.status_code == 200  # routed to OpenAI base and forwarded

    async def test_openai_chat_completions_compresses(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False)
        body = json.dumps(
            {"model": "gpt", "messages": [{"role": "user", "content": "hi   \n\n\n\n"}]}
        ).encode()
        await eng.handle("POST", "/v1/chat/completions", [("authorization", "Bearer x")], body)
        sent = json.loads(up.last.content)
        assert sent["messages"][0]["content"] == "hi\n\n"


class TestMetrics:
    async def test_emits_one_savings_event_per_request(self) -> None:
        spy = SpySink()
        eng = _engine(FakeUpstream(), cache_enabled=False, metrics=spy)
        await eng.handle(
            "POST",
            "/v1/messages",
            [("x-request-id", "abc"), ("x-api-key", "k")],
            _anthropic("hello   \n\n\n\n"),
        )
        assert len(spy.events) == 1
        event = spy.events[0]
        assert event.request_id == "abc"  # correlation id taken from the header
        assert event.dialect == "anthropic"
        assert event.canonicalized is True
        assert event.tokens_before >= event.tokens_after
        assert event.status_code == 200
        assert event.duration_ms >= 0.0

    async def test_generates_request_id_when_header_absent(self) -> None:
        spy = SpySink()
        eng = _engine(FakeUpstream(), metrics=spy)
        await eng.handle("GET", "/v1/models", [], b"")  # unroutable -> 502, still recorded
        assert len(spy.events) == 1
        assert spy.events[0].status_code == 502
        assert spy.events[0].request_id  # a non-empty generated id

    async def test_event_carries_tenant_in_multi_tenant_mode(self) -> None:
        from parcus.tenant import derive_tenant

        spy = SpySink()
        eng = _engine(FakeUpstream(), cache_enabled=False, multi_tenant=True, metrics=spy)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("hi"))
        assert spy.events[0].tenant == derive_tenant([("x-api-key", "k")])

    async def test_event_has_empty_tenant_in_single_tenant_mode(self) -> None:
        spy = SpySink()
        eng = _engine(FakeUpstream(), cache_enabled=False, metrics=spy)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("hi"))
        assert spy.events[0].tenant == ""


class TestMultiTenantIsolation:
    """Hosted mode: the cache is namespaced per credential-derived tenant (BOLA defence)."""

    async def test_different_tenants_never_share_cache(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=True)
        body = _anthropic("identical body across tenants")
        # Tenant A primes the cache, then tenant B sends the SAME body with a different key.
        a = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        b = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-b")], body)
        assert a.meta["cache"] == "miss"
        assert b.meta["cache"] == "miss"  # NOT served from tenant A's entry
        assert up.calls == 2  # both reached upstream — no cross-tenant leak

    async def test_same_tenant_still_hits_cache(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=True)
        body = _anthropic("same tenant repeats a request")
        first = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        assert first.meta["cache"] == "miss"
        assert second.meta["cache"] == "hit"
        assert up.calls == 1

    async def test_single_tenant_mode_unaffected(self) -> None:
        # With multi_tenant off (default), the credential does not scope the cache: a repeated
        # body hits regardless of key — correct for the local single-principal deployment.
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache(), multi_tenant=False)
        body = _anthropic("local single user")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k1")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "k2")], body)
        assert second.meta["cache"] == "hit"
        assert up.calls == 1


class TestEdgeAuthorization:
    """Hosted mode: an optional allow-list authorizes callers before forwarding (fail closed)."""

    async def test_listed_tenant_is_forwarded(self) -> None:
        from parcus.tenant import derive_tenant

        up = FakeUpstream()
        allowed = frozenset({derive_tenant([("x-api-key", "good-key")])})
        eng = _engine(up, cache_enabled=False, multi_tenant=True, allowed_tenants=allowed)
        result = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "good-key")], _anthropic("hi")
        )
        assert result.status_code == 200
        assert up.calls == 1

    async def test_unlisted_tenant_gets_401_without_upstream(self) -> None:
        from parcus.tenant import derive_tenant

        up = FakeUpstream()
        allowed = frozenset({derive_tenant([("x-api-key", "good-key")])})
        eng = _engine(up, cache_enabled=False, multi_tenant=True, allowed_tenants=allowed)
        result = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "bad-key")], _anthropic("hi")
        )
        assert result.status_code == 401
        assert result.meta["auth"] == "denied"
        assert up.calls == 0  # never reached the provider

    async def test_anonymous_request_denied_when_allow_list_set(self) -> None:
        # No credential header -> anonymous -> not in the allow-list -> 401 (still routable via
        # anthropic-version), proving fail-closed behaviour.
        up = FakeUpstream()
        eng = _engine(
            up, cache_enabled=False, multi_tenant=True, allowed_tenants=frozenset({"abc"})
        )
        result = await eng.handle(
            "POST", "/v1/messages", [("anthropic-version", "2023-06-01")], _anthropic("hi")
        )
        assert result.status_code == 401
        assert up.calls == 0

    async def test_empty_allow_list_forwards_everything(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache_enabled=False, multi_tenant=True)  # no allow-list
        result = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "any-key")], _anthropic("hi")
        )
        assert result.status_code == 200
        assert up.calls == 1


class TestRateLimiting:
    """A per-tenant rate limiter sheds over-limit requests with 429 before any upstream call."""

    async def test_over_limit_request_gets_429_with_retry_after(self) -> None:
        from parcus.quota import RateLimit, RateLimiter

        up = FakeUpstream()
        limiter = RateLimiter(RateLimit(capacity=1, refill_per_sec=1.0))
        eng = _engine(up, cache_enabled=False, rate_limiter=limiter)
        body = _anthropic("hi")
        first = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        second = await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.meta["rate"] == "limited"
        assert dict(second.headers).get("retry-after") == "1"
        assert up.calls == 1  # the limited request never reached the provider

    async def test_rate_limit_is_per_tenant(self) -> None:
        from parcus.quota import RateLimit, RateLimiter

        up = FakeUpstream()
        limiter = RateLimiter(RateLimit(capacity=1, refill_per_sec=1.0))
        eng = _engine(up, cache_enabled=False, multi_tenant=True, rate_limiter=limiter)
        body = _anthropic("hi")
        await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        a2 = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-a")], body)
        b1 = await eng.handle("POST", "/v1/messages", [("x-api-key", "tenant-b")], body)
        assert a2.status_code == 429  # tenant a exhausted its bucket
        assert b1.status_code == 200  # tenant b has its own bucket


class _MarkerEmbedder:
    """Deterministic embedder keyed on a marker token in the request text."""

    def embed(self, texts: object) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:  # type: ignore[attr-defined]
            if "ALPHA" in t:
                out.append([1.0, 0.0])
            elif "BETA" in t:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out


class TestSimilarityCache:
    """Opt-in semantic cache: serve a near-duplicate's response on an exact miss."""

    async def test_near_duplicate_served_without_upstream(self) -> None:
        from parcus.cache import SimilarityCache

        up = FakeUpstream()
        sim = SimilarityCache(_MarkerEmbedder(), threshold=0.97)
        eng = _engine(up, cache=SqliteCache(), similarity=sim)
        first = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "k")], _anthropic("ALPHA one")
        )
        second = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "k")], _anthropic("ALPHA two")
        )
        assert first.meta["cache"] == "miss"
        assert second.meta["cache"] == "similar"  # near-duplicate of the first
        assert up.calls == 1  # the second request never reached the provider
        assert second.content == first.content

    async def test_dissimilar_request_forwards(self) -> None:
        from parcus.cache import SimilarityCache

        up = FakeUpstream()
        sim = SimilarityCache(_MarkerEmbedder(), threshold=0.97)
        eng = _engine(up, cache=SqliteCache(), similarity=sim)
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("ALPHA one"))
        other = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "k")], _anthropic("BETA two")
        )
        assert other.meta["cache"] == "miss"
        assert up.calls == 2  # dissimilar -> forwarded

    async def test_disabled_by_default(self) -> None:
        up = FakeUpstream()
        eng = _engine(up, cache=SqliteCache())  # no similarity injected
        await eng.handle("POST", "/v1/messages", [("x-api-key", "k")], _anthropic("ALPHA one"))
        second = await eng.handle(
            "POST", "/v1/messages", [("x-api-key", "k")], _anthropic("ALPHA two")
        )
        assert second.meta["cache"] == "miss"
        assert up.calls == 2
