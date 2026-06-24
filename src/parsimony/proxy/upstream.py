"""The outbound provider port and its httpx adapter (non-streaming).

This port is part of the imperative shell, not the core, so it lives beside its adapter rather
than in :mod:`parsimony.ports`. Streaming is handled separately in the app layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

__all__ = ["HttpxUpstream", "UpstreamPort", "UpstreamRequest", "UpstreamResponse"]

Headers = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class UpstreamRequest:
    """A buffered request to forward to a provider."""

    method: str
    url: str
    headers: Headers
    content: bytes


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    """A buffered provider response."""

    status_code: int
    headers: Headers
    content: bytes


@runtime_checkable
class UpstreamPort(Protocol):
    """Forwards a buffered request to the real provider and returns its response."""

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        """Send ``request`` upstream and return the buffered response."""
        ...


class HttpxUpstream:
    """An :class:`UpstreamPort` backed by a shared :class:`httpx.AsyncClient`.

    Args:
        timeout_seconds: Read timeout for the upstream call (LLM responses can be slow).
    """

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        """Create the adapter with a generous read timeout suited to LLM latency."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            follow_redirects=False,  # never follow redirects to other hosts (SSRF)
        )

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        """Forward the request and buffer the response."""
        response = await self._client.request(
            request.method,
            request.url,
            headers=list(request.headers),
            content=request.content,
        )
        return UpstreamResponse(
            status_code=response.status_code,
            headers=tuple(response.headers.items()),
            content=response.content,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
