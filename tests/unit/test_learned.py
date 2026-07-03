"""Tests for the Tier-2 learned compressor (span handling via a fake local reducer)."""

from __future__ import annotations

import math

from parcus.compress.learned import (
    DEFAULT_LLMLINGUA2_MODEL,
    DEFAULT_LLMLINGUA_MODEL,
    LearnedCompressor,
    LLMLinguaReducer,
)
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*spans: Span, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=spans),),
        system=system,
    )


class _HalfReducer:
    """Deterministic stand-in: keep the first ``ceil(keep_ratio * n)`` whitespace tokens."""

    def reduce(self, text: str, *, keep_ratio: float) -> str:
        words = text.split()
        if not words:
            return text
        keep = max(1, math.ceil(len(words) * keep_ratio))
        return " ".join(words[:keep])


class _BoomReducer:
    def reduce(self, text: str, *, keep_ratio: float) -> str:
        raise RuntimeError("model exploded")


class _NoopReducer:
    def reduce(self, text: str, *, keep_ratio: float) -> str:
        return text


def test_learned_preserves_immutable_blocks_in_structured_message() -> None:
    # A structured message with only immutable blocks (tool_result) must be returned verbatim.
    raw = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]}
    req = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(
            Message(role=Role.ASSISTANT, spans=(Span("one two three four"),)),
            Message(role=Role.USER, spans=(), raw=raw),
        ),
    )
    out, _ = LearnedCompressor(_HalfReducer(), keep_ratio=0.5).compress(req)
    assert out.messages[1] is req.messages[1]  # no text block to reduce -> same object
    assert out.messages[1].raw == raw
    assert out.messages[0].text == "one two"  # ceil(4*0.5)=2 kept


def test_learned_reduces_text_blocks_inside_structured_message() -> None:
    # text blocks inside a structured message are reduced; sibling immutable blocks stay verbatim.
    raw = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "one two three four"},
            {"type": "tool_use", "id": "t", "name": "sh", "input": {"cmd": "ls"}},
        ],
    }
    req = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.ASSISTANT, spans=(), raw=raw),),
    )
    out, stats = LearnedCompressor(_HalfReducer(), keep_ratio=0.5).compress(req)
    new_raw = out.messages[0].raw
    assert new_raw is not None
    assert new_raw["content"][0]["text"] == "one two"  # ceil(4*0.5)=2 kept
    assert new_raw["content"][1] == raw["content"][1]  # tool_use block untouched
    assert stats[0].spans_touched == 1


def test_learned_leaves_structured_message_untouched_when_reduce_is_noop() -> None:
    # A text block that the reducer doesn't shrink yields the same message object (touched=0).
    raw = {"role": "assistant", "content": [{"type": "text", "text": "keep"}]}
    req = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.ASSISTANT, spans=(), raw=raw),),
    )
    out, stats = LearnedCompressor(_NoopReducer()).compress(req)
    assert out.messages[0] is req.messages[0]  # unchanged -> same object
    assert stats[0].spans_touched == 0


def test_learned_leaves_structured_message_with_non_list_content_verbatim() -> None:
    # A raw message whose content isn't a block list (no text blocks to reach) is untouched.
    raw = {"role": "user", "content": "plain string body"}
    req = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=(), raw=raw),),
    )
    out, stats = LearnedCompressor(_HalfReducer(), keep_ratio=0.5).compress(req)
    assert out.messages[0] is req.messages[0]  # same object -> verbatim
    assert stats[0].spans_touched == 0


class TestLLMLinguaReducerBackend:
    """Backend selection + default model resolution (no model load — construction only)."""

    def test_v1_is_the_default_backend(self) -> None:
        assert LLMLinguaReducer().model_name == DEFAULT_LLMLINGUA_MODEL

    def test_llmlingua2_selects_its_default_model(self) -> None:
        assert LLMLinguaReducer(use_llmlingua2=True).model_name == DEFAULT_LLMLINGUA2_MODEL

    def test_explicit_model_overrides_the_backend_default(self) -> None:
        assert LLMLinguaReducer("my/local-model").model_name == "my/local-model"
        # Even with the v2 backend selected, an explicit name wins.
        v2 = LLMLinguaReducer("my/local-model", use_llmlingua2=True)
        assert v2.model_name == "my/local-model"

    def test_empty_model_name_falls_back_to_backend_default(self) -> None:
        assert LLMLinguaReducer("").model_name == DEFAULT_LLMLINGUA_MODEL
        assert LLMLinguaReducer("", use_llmlingua2=True).model_name == DEFAULT_LLMLINGUA2_MODEL


class TestLearnedCompressor:
    def test_reduces_mutable_prose(self) -> None:
        comp = LearnedCompressor(_HalfReducer(), keep_ratio=0.5)
        out, stats = comp.compress(_req(Span("one two three four five six")))
        assert out.messages[0].text == "one two three"  # ceil(6*0.5)=3 kept
        assert stats[0].step == "learned"
        assert stats[0].spans_touched == 1
        assert stats[0].tokens_after <= stats[0].tokens_before

    def test_reports_no_runtime_invariant(self) -> None:
        # Tier-2 has no model-free invariant -> ok is None (accuracy comes from the offline gate).
        _out, stats = LearnedCompressor(_HalfReducer()).compress(_req(Span("a b c d")))
        assert stats[0].ok is None

    def test_never_touches_immutable_spans(self) -> None:
        comp = LearnedCompressor(_HalfReducer(), keep_ratio=0.5)
        out, stats = comp.compress(_req(Span("alpha beta gamma delta", mutable=False)))
        assert out.messages[0].spans[0].text == "alpha beta gamma delta"
        assert stats[0].spans_touched == 0

    def test_reduces_system_prose_but_preserves_code(self) -> None:
        comp = LearnedCompressor(_HalfReducer(), keep_ratio=0.5)
        out, _stats = comp.compress(_req(Span("hi"), system="aa bb cc dd ```keep all code here```"))
        assert "```keep all code here```" in (out.system or "")  # fenced code untouched
        assert "aa bb" in (out.system or "")  # prose reduced

    def test_noop_reducer_leaves_request_unchanged(self) -> None:
        out, stats = LearnedCompressor(_NoopReducer()).compress(
            _req(Span("nothing to cut"), system="also unchanged")
        )
        assert out.messages[0].text == "nothing to cut"
        assert out.system == "also unchanged"  # system present but reduce is a no-op
        assert stats[0].spans_touched == 0

    def test_fails_open_when_reducer_raises(self) -> None:
        request = _req(Span("this will explode"))
        out, stats = LearnedCompressor(_BoomReducer()).compress(request)
        assert out is request  # original returned unchanged
        assert stats == ()  # no stats on the fail-open path
