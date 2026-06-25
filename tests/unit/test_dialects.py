"""Tests for dialect detection, parsing, and serialisation (the conservative M1 subset)."""

from __future__ import annotations

from parcus.model import Dialect, Role
from parcus.proxy.dialects import detect, parse, serialize


class TestDetect:
    def test_anthropic(self) -> None:
        assert detect("/v1/messages") is Dialect.ANTHROPIC

    def test_openai(self) -> None:
        assert detect("/v1/chat/completions") is Dialect.OPENAI

    def test_unknown(self) -> None:
        assert detect("/v1/models") is Dialect.UNKNOWN


class TestParseAnthropic:
    def test_simple_text_request(self) -> None:
        body = {
            "model": "claude-x",
            "system": "be terse",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "t"}],
        }
        req = parse(Dialect.ANTHROPIC, body)
        assert req is not None
        assert req.model == "claude-x"
        assert req.system == "be terse"
        assert req.messages[0].role is Role.USER
        assert req.messages[0].text == "hello"
        assert req.tools_json is not None

    def test_rejects_block_content(self) -> None:
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        assert parse(Dialect.ANTHROPIC, body) is None

    def test_rejects_non_string_system(self) -> None:
        body = {"system": [{"type": "text", "text": "x"}], "messages": []}
        assert parse(Dialect.ANTHROPIC, body) is None

    def test_rejects_unknown_role(self) -> None:
        body = {"messages": [{"role": "system", "content": "x"}]}  # system not an Anthropic role
        assert parse(Dialect.ANTHROPIC, body) is None

    def test_rejects_non_list_messages(self) -> None:
        assert parse(Dialect.ANTHROPIC, {"messages": "nope"}) is None


class TestParseOpenAI:
    def test_simple_text_request(self) -> None:
        body = {
            "model": "gpt-x",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        }
        req = parse(Dialect.OPENAI, body)
        assert req is not None
        assert [m.role for m in req.messages] == [Role.SYSTEM, Role.USER]

    def test_rejects_extra_message_keys(self) -> None:
        body = {"messages": [{"role": "assistant", "content": "x", "tool_calls": []}]}
        assert parse(Dialect.OPENAI, body) is None

    def test_rejects_tool_role(self) -> None:
        body = {"messages": [{"role": "tool", "content": "x"}]}
        assert parse(Dialect.OPENAI, body) is None

    def test_rejects_non_string_content(self) -> None:
        body = {"messages": [{"role": "user", "content": 5}]}
        assert parse(Dialect.OPENAI, body) is None


class TestUnknownDialect:
    def test_returns_none(self) -> None:
        assert parse(Dialect.UNKNOWN, {"messages": []}) is None


class TestSerialize:
    def test_anthropic_roundtrip_preserves_fields(self) -> None:
        body = {"model": "claude-x", "max_tokens": 10, "system": "old", "messages": []}
        req = parse(
            Dialect.ANTHROPIC, {"system": "new", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert req is not None
        out = serialize(req, body)
        assert out["model"] == "claude-x"  # preserved from original
        assert out["max_tokens"] == 10  # preserved
        assert out["system"] == "new"  # from canonical
        assert out["messages"] == [{"role": "user", "content": "hi"}]

    def test_openai_serialises_messages(self) -> None:
        req = parse(Dialect.OPENAI, {"messages": [{"role": "user", "content": "hi"}]})
        assert req is not None
        out = serialize(req, {"model": "gpt-x", "messages": []})
        assert out["messages"] == [{"role": "user", "content": "hi"}]
        assert out["model"] == "gpt-x"
