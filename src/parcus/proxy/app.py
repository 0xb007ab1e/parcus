"""FastAPI ingress: a catch-all reverse proxy in front of the engine.

Non-streaming requests go through :class:`~parcus.proxy.engine.ProxyEngine` (compress +
cache + forward). Streaming requests are transparently passed through (no compression/cache in
M1 — that is an M2 enhancement); they still route to the correct provider and stream the
response back untouched. Everything fails open.
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
    _DROP_REQUEST_HEADERS,
    _DROP_RESPONSE_HEADERS,
    ProxyEngine,
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


def _is_stream(body: bytes) -> bool:
    """Return whether the request body opts into a streaming response."""
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
    return out


async def _stream_passthrough(
    app: FastAPI,
    engine: ProxyEngine,
    method: str,
    path: str,
    headers: list[tuple[str, str]],
    body: bytes,
) -> Response:
    """Transparently proxy a streaming request/response (no compression/cache in M1)."""
    base = engine.route(detect(path), headers)
    if base is None:
        return JSONResponse(
            status_code=502,
            content={"error": "parcus: unable to route request to a provider"},
        )
    url = base.rstrip("/") + path
    forward = [(k, v) for k, v in headers if k.lower() not in _DROP_REQUEST_HEADERS]
    client: httpx.AsyncClient = app.state.stream_client
    upstream = await client.send(
        client.build_request(method, url, headers=forward, content=body), stream=True
    )
    response_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS
    }
    response_headers["x-parcus-cache"] = "stream-bypass"
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
        if _is_stream(body):
            return await _stream_passthrough(
                request.app, engine, request.method, request.url.path, headers, body
            )
        result = await engine.handle(request.method, request.url.path, headers, body)
        response = Response(
            content=result.content,
            status_code=result.status_code,
            headers=dict(result.headers),
        )
        for key, value in _meta_headers(result.meta).items():
            response.headers[key] = value
        return response

    return app
