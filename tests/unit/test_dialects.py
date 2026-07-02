"""Tests for dialect detection, parsing, and serialisation (the conservative M1 subset)."""

from __future__ import annotations

from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
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

    def test_no_breakpoint_serialises_plain_strings(self) -> None:
        # An un-annotated request keeps the plain-string content form (byte-identical to before).
        req = parse(
            Dialect.ANTHROPIC,
            {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        )
        assert req is not None
        out = serialize(req, {"messages": []})
        assert out["messages"] == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]


class TestStructuredRoundTrip:
    """M1d slice 1: structured messages are carried verbatim and round-trip structurally exact."""

    def _assert_roundtrips(self, dialect: Dialect, body: dict[str, object]) -> None:
        req = parse(dialect, body, structured=True)
        assert req is not None
        assert serialize(req, body) == body  # parse -> serialize identity (modulo separators)

    def test_anthropic_tool_use_and_result(self) -> None:
        self._assert_roundtrips(
            Dialect.ANTHROPIC,
            {
                "model": "claude-x",
                "messages": [
                    {"role": "user", "content": "run it"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {
                                "type": "tool_use",
                                "id": "tu_1",
                                "name": "sh",
                                "input": {"cmd": "ls"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "tu_1", "content": "a\nb"},
                        ],
                    },
                ],
            },
        )

    def test_anthropic_image_block(self) -> None:
        self._assert_roundtrips(
            Dialect.ANTHROPIC,
            {
                "model": "claude-x",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBOR",
                                },
                            },
                            {"type": "text", "text": "what is this"},
                        ],
                    }
                ],
            },
        )

    def test_openai_tool_calls_and_tool_role(self) -> None:
        self._assert_roundtrips(
            Dialect.OPENAI,
            {
                "model": "gpt-x",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "w", "arguments": "{}"},
                            },
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                ],
            },
        )

    def test_structured_off_passes_through_as_none(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        }
        assert parse(Dialect.ANTHROPIC, body) is None  # structured defaults False → unhandled

    def test_unknown_role_is_none_even_structured(self) -> None:
        body = {"model": "m", "messages": [{"role": "developer", "content": "x"}]}
        assert parse(Dialect.OPENAI, body, structured=True) is None  # unknown role → passthrough

    def test_anthropic_block_system_is_none_even_structured(self) -> None:
        body = {
            "model": "m",
            "system": [{"type": "text", "text": "sys"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert parse(Dialect.ANTHROPIC, body, structured=True) is None  # block-list system deferred


class TestSerializeCacheBreakpoint:
    def _req(
        self, dialect: Dialect, texts: tuple[str, ...], cache_breakpoint: int | None
    ) -> CanonicalRequest:
        return CanonicalRequest(
            dialect=dialect,
            model="m",
            messages=tuple(Message(role=Role.USER, spans=(Span(t),)) for t in texts),
            cache_breakpoint=cache_breakpoint,
        )

    def test_anthropic_renders_cache_control_at_marked_message(self) -> None:
        req = self._req(Dialect.ANTHROPIC, ("stable", "volatile"), cache_breakpoint=0)
        out = serialize(req, {"messages": []})
        assert out["messages"][0] == {
            "role": "user",
            "content": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}],
        }
        # Only the marked message is expanded; the rest stay plain strings.
        assert out["messages"][1] == {"role": "user", "content": "volatile"}

    def test_out_of_range_breakpoint_is_ignored(self) -> None:
        req = self._req(Dialect.ANTHROPIC, ("a", "b"), cache_breakpoint=99)
        out = serialize(req, {"messages": []})
        assert out["messages"] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]

    def test_openai_ignores_breakpoint(self) -> None:
        # cache_control is Anthropic-specific; an OpenAI request never renders it.
        req = self._req(Dialect.OPENAI, ("a", "b"), cache_breakpoint=0)
        out = serialize(req, {"messages": []})
        assert out["messages"] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
