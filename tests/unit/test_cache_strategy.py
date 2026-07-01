"""Unit tests for the per-provider cache strategies and the dialect registry.

Covers the M1a slice: the ``CacheStrategy`` port's two implementations (``NullCacheStrategy``,
``AnthropicCacheStrategy``), their capability descriptors, the cacheable-prefix boundary logic,
the fail-open identity ``annotate``, and the ``cache_strategy`` registry fallback.
"""

from __future__ import annotations

from parcus.cache.strategy import (
    AnthropicCacheStrategy,
    NullCacheStrategy,
    cache_strategy,
)
from parcus.model import (
    CacheCapability,
    CacheModel,
    CanonicalRequest,
    Dialect,
    Message,
    Role,
    Span,
)
from parcus.ports import CacheStrategy


def _msg(text: str, role: Role = Role.USER) -> Message:
    return Message(role=role, spans=(Span(text),))


def _req(
    *,
    dialect: Dialect = Dialect.ANTHROPIC,
    messages: tuple[Message, ...] = (),
    system: str | None = None,
    tools_json: str | None = None,
) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=dialect,
        model="m",
        messages=messages,
        system=system,
        tools_json=tools_json,
    )


# --- CacheModel / CacheCapability data --------------------------------------------------------


def test_cache_model_members() -> None:
    assert CacheModel.NONE == "none"
    assert CacheModel.AUTOMATIC_PREFIX == "automatic_prefix"
    assert CacheModel.EXPLICIT_BREAKPOINT == "explicit_breakpoint"


def test_cache_capability_defaults() -> None:
    cap = CacheCapability(model=CacheModel.NONE)
    assert cap.model is CacheModel.NONE
    assert cap.min_prefix_tokens == 0
    assert cap.max_breakpoints == 0


# --- NullCacheStrategy ------------------------------------------------------------------------


class TestNullCacheStrategy:
    def test_capability_is_none_model(self) -> None:
        cap = NullCacheStrategy().capability
        assert cap.model is CacheModel.NONE
        assert cap.min_prefix_tokens == 0
        assert cap.max_breakpoints == 0

    def test_boundary_is_always_none(self) -> None:
        strat = NullCacheStrategy()
        assert strat.cacheable_boundary(_req()) is None
        assert strat.cacheable_boundary(_req(system="you are helpful")) is None
        assert strat.cacheable_boundary(_req(messages=(_msg("a"), _msg("b")))) is None

    def test_annotate_is_identity(self) -> None:
        req = _req(messages=(_msg("hi"),))
        assert NullCacheStrategy().annotate(req) is req


# --- AnthropicCacheStrategy -------------------------------------------------------------------


class TestAnthropicCacheStrategy:
    def test_capability(self) -> None:
        cap = AnthropicCacheStrategy().capability
        assert cap.model is CacheModel.EXPLICIT_BREAKPOINT
        assert cap.min_prefix_tokens == 4096  # conservative floor across Anthropic models
        assert cap.max_breakpoints == 4

    def test_boundary_protects_all_but_last_turn(self) -> None:
        strat = AnthropicCacheStrategy()
        three = (_msg("sys ctx"), _msg("prior"), _msg("the question"))
        assert strat.cacheable_boundary(_req(messages=three)) == 2

    def test_boundary_single_message_protects_prefix_only(self) -> None:
        strat = AnthropicCacheStrategy()
        assert strat.cacheable_boundary(_req(messages=(_msg("only turn"),))) == 0

    def test_boundary_no_messages_with_system(self) -> None:
        strat = AnthropicCacheStrategy()
        assert strat.cacheable_boundary(_req(system="stable system prompt")) == 0

    def test_boundary_no_messages_with_tools_only(self) -> None:
        strat = AnthropicCacheStrategy()
        assert strat.cacheable_boundary(_req(tools_json='[{"name":"read"}]')) == 0

    def test_boundary_empty_request_is_none(self) -> None:
        strat = AnthropicCacheStrategy()
        assert strat.cacheable_boundary(_req()) is None

    def test_annotate_is_identity_pending_m1b(self) -> None:
        req = _req(messages=(_msg("a"), _msg("b")), system="s")
        assert AnthropicCacheStrategy().annotate(req) is req


# --- Registry ---------------------------------------------------------------------------------


class TestCacheStrategyRegistry:
    def test_anthropic_resolves_to_anthropic_strategy(self) -> None:
        assert isinstance(cache_strategy(Dialect.ANTHROPIC), AnthropicCacheStrategy)

    def test_openai_falls_back_to_null(self) -> None:
        # OpenAI is automatic-prefix (preserve-only); its dedicated strategy is a later slice,
        # so it falls back to the cache-neutral Null strategy for now.
        assert isinstance(cache_strategy(Dialect.OPENAI), NullCacheStrategy)

    def test_unknown_falls_back_to_null(self) -> None:
        assert isinstance(cache_strategy(Dialect.UNKNOWN), NullCacheStrategy)


# --- Structural conformance to the port -------------------------------------------------------


def test_strategies_satisfy_the_port() -> None:
    assert isinstance(NullCacheStrategy(), CacheStrategy)
    assert isinstance(AnthropicCacheStrategy(), CacheStrategy)
