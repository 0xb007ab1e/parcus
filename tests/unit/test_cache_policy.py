"""Tests for the cache eligibility policy."""

from __future__ import annotations

from parsimony.cache import CachePolicy
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(text: str) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.OPENAI,
        model="m",
        messages=(Message(role=Role.USER, spans=(Span(text),)),),
    )


class TestShouldCache:
    def test_default_allows_normal_request(self) -> None:
        assert CachePolicy().should_cache(_req("write a haiku")) is True

    def test_disabled_blocks_everything(self) -> None:
        assert CachePolicy(enabled=False).should_cache(_req("anything")) is False

    def test_nocache_pattern_blocks(self) -> None:
        policy = CachePolicy.from_patterns([r"/auth/login"])
        assert policy.should_cache(_req("POST /auth/login")) is False
        assert policy.should_cache(_req("GET /health")) is True

    def test_secret_bypass_blocks_when_detector_positive(self) -> None:
        policy = CachePolicy(bypass_on_secret=True)
        assert policy.should_cache(_req("k"), has_secret=lambda _t: True) is False

    def test_secret_bypass_requires_a_detector(self) -> None:
        # With no detector supplied we cannot bypass; eligibility falls through to True.
        assert CachePolicy(bypass_on_secret=True).should_cache(_req("k")) is True

    def test_secret_bypass_disabled_allows_even_with_secret(self) -> None:
        policy = CachePolicy(bypass_on_secret=False)
        assert policy.should_cache(_req("k"), has_secret=lambda _t: True) is True
