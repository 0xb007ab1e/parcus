"""Tests for deterministic cache-key derivation."""

from __future__ import annotations

from parcus.cache import compute_key
from parcus.cache.key import KEY_VERSION
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(text: str, *, model: str = "m", stream: bool = False) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model=model,
        messages=(Message(role=Role.USER, spans=(Span(text),)),),
        stream=stream,
    )


class TestComputeKey:
    def test_is_deterministic(self) -> None:
        assert compute_key(_req("hello")) == compute_key(_req("hello"))

    def test_differs_on_content(self) -> None:
        assert compute_key(_req("hello")) != compute_key(_req("hello!"))

    def test_differs_on_model(self) -> None:
        assert compute_key(_req("hi", model="a")) != compute_key(_req("hi", model="b"))

    def test_ignores_stream_flag(self) -> None:
        # Streaming vs not yields the same logical content, so it must not change the key.
        assert compute_key(_req("hi", stream=True)) == compute_key(_req("hi", stream=False))

    def test_salt_changes_key(self) -> None:
        assert compute_key(_req("hi"), salt="A") != compute_key(_req("hi"), salt="B")
        assert compute_key(_req("hi")) != compute_key(_req("hi"), salt="A")

    def test_has_version_prefix(self) -> None:
        assert compute_key(_req("hi")).startswith(f"{KEY_VERSION}:")
