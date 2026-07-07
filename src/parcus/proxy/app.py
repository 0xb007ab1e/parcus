"""FastAPI ingress: a catch-all reverse proxy in front of the engine.

Non-streaming requests go through :class:`~parcus.proxy.engine.ProxyEngine` (compress +
cache + forward). Streaming requests have their **request body compressed** (and are authorized
+ rate-limited) via :meth:`ProxyEngine.prepare_stream`, then forwarded; the SSE **response** is
relayed back byte-for-byte and unbuffered. They are not response-cached (streams aren't cached).
Everything fails open.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from parcus import __version__
from parcus.obs import render_prometheus
from parcus.proxy.dialects import detect
from parcus.proxy.engine import (
    _DROP_RESPONSE_HEADERS,
    ProxyEngine,
    ProxyResult,
)

# Reserved local paths the proxy answers itself (never forwarded upstream).
_STATS_PATH = "/__parcus__/stats"
_HEALTH_PATH = "/__parcus__/health"
_METRICS_PATH = "/__parcus__/metrics"


@runtime_checkable
class StatsSource(Protocol):
    """Anything that can produce an aggregate metrics snapshot for the stats endpoint."""

    def snapshot(self) -> dict[str, Any]:
        """Return the aggregate metrics snapshot."""
        ...


__all__ = ["create_app"]

_STREAM_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def _streams_by_path(path: str) -> bool:
    """Return whether ``path`` is a streaming endpoint identified by its suffix.

    Some providers select streaming by the endpoint rather than a body flag — Gemini's
    ``…:streamGenerateContent`` (paired with ``?alt=sse``). Recognising it here keeps such
    requests on the streaming path so their response is relayed unbuffered.
    """
    return path.endswith(":streamGenerateContent")


def _is_stream(path: str, body: bytes) -> bool:
    """Return whether the request opts into a streaming response, by path or body flag.

    Either signal ⇒ stream: a streaming endpoint path (Gemini) or an explicit ``stream: true`` in
    the JSON body (Anthropic/OpenAI). Erring toward "stream" is the safe choice — buffering a
    response the client expects to stream would break the harness.
    """
    if _streams_by_path(path):
        return True
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(decoded, dict) and bool(decoded.get("stream"))


def _meta_headers(meta: dict[str, Any]) -> dict[str, str]:
    """Render engine metadata as ``X-Parcus-*`` response headers."""
    out: dict[str, str] = {}
    if "cache" in meta:
        out["x-parcus-cache"] = str(meta["cache"])
    if "dialect" in meta:
        out["x-parcus-dialect"] = str(meta["dialect"])
    if "memory" in meta:
        out["x-parcus-memory"] = str(meta["memory"])
    if "tokens_before" in meta and "tokens_after" in meta:
        before, after = int(meta["tokens_before"]), int(meta["tokens_after"])
        out["x-parcus-tokens-before"] = str(before)
        out["x-parcus-tokens-after"] = str(after)
        out["x-parcus-tokens-saved"] = str(max(0, before - after))
    usage = meta.get("upstream_usage")
    if usage is not None:
        # Provider's billed counts + prompt-cache signal (content-free); cache-read > 0 means the
        # re-sent prefix hit the provider's prompt cache — watch it to confirm compression didn't
        # bust that cache (PLAN Q3).
        out["x-parcus-upstream-input-tokens"] = str(usage.input_tokens)
        out["x-parcus-upstream-output-tokens"] = str(usage.output_tokens)
        out["x-parcus-upstream-cache-read-tokens"] = str(usage.cache_read_tokens)
        out["x-parcus-upstream-cache-write-tokens"] = str(usage.cache_write_tokens)
    return out


def _result_to_response(result: ProxyResult) -> Response:
    """Turn a buffered :class:`ProxyResult` into a Response with its ``x-parcus-*`` headers."""
    response = Response(
        content=result.content,
        status_code=result.status_code,
        headers=dict(result.headers),
    )
    for key, value in _meta_headers(result.meta).items():
        response.headers[key] = value
    return response


async def _stream_passthrough(
    app: FastAPI,
    engine: ProxyEngine,
    method: str,
    path: str,
    headers: list[tuple[str, str]],
    body: bytes,
    query: str = "",
) -> Response:
    """Proxy a streaming request: forward the COMPRESSED request body, stream the response back.

    The request payload is compressed (and authorized + rate-limited) via
    :meth:`ProxyEngine.prepare_stream`; the SSE response is relayed byte-for-byte and unbuffered.
    ``query`` is forwarded verbatim so streaming selectors like Gemini's ``?alt=sse`` survive.
    """
    plan = engine.prepare_stream(detect(path), path, headers, body, query=query)
    if plan.early is not None:
        return _result_to_response(plan.early)
    client: httpx.AsyncClient = app.state.stream_client
    upstream = await client.send(
        client.build_request(
            method, plan.url, headers=list(plan.forward_headers), content=plan.out_body
        ),
        stream=True,
    )
    response_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS
    }
    response_headers.update(_meta_headers(plan.meta))  # x-parcus-cache=stream-bypass + token counts
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream.aclose),
    )


def create_app(
    engine: ProxyEngine,
    *,
    stream_client: httpx.AsyncClient | None = None,
    stats_source: StatsSource | None = None,
) -> FastAPI:
    """Build the proxy ASGI app around an injected engine.

    Args:
        engine: The orchestration engine (its upstream adapter handles non-streaming calls).
        stream_client: Optional injected httpx client for the streaming path (tests); when
            omitted one is created and owned by the app lifespan.
        stats_source: Optional metrics source; when given, ``GET /__parcus__/stats`` returns
            its JSON snapshot (handled locally, never forwarded upstream).

    Returns:
        A configured :class:`fastapi.FastAPI` application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = stream_client or httpx.AsyncClient(timeout=_STREAM_TIMEOUT, follow_redirects=False)
        app.state.stream_client = client
        try:
            yield
        finally:
            if stream_client is None:
                await client.aclose()

    app = FastAPI(title="parcus", lifespan=lifespan)

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(request: Request) -> Response:
        if request.url.path == _HEALTH_PATH:
            return JSONResponse({"status": "ok", "version": __version__})
        if request.url.path == _STATS_PATH:
            snapshot = stats_source.snapshot() if stats_source is not None else {}
            return JSONResponse(snapshot)
        if request.url.path == _METRICS_PATH:
            text = render_prometheus(stats_source.snapshot()) if stats_source is not None else ""
            return PlainTextResponse(text, media_type="text/plain; version=0.0.4")
        body = await request.body()
        headers = list(request.headers.items())
        query = request.url.query
        if _is_stream(request.url.path, body):
            return await _stream_passthrough(
                request.app, engine, request.method, request.url.path, headers, body, query
            )
        result = await engine.handle(request.method, request.url.path, headers, body, query=query)
        return _result_to_response(result)

    return app
