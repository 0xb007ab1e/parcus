"""Tests for dialect detection, parsing, and serialisation (the conservative M1 subset)."""

from __future__ import annotations

from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
from parcus.proxy.dialects import detect, gemini_model_from_path, parse, serialize


class TestDetect:
    def test_anthropic(self) -> None:
        assert detect("/v1/messages") is Dialect.ANTHROPIC

    def test_openai(self) -> None:
        assert detect("/v1/chat/completions") is Dialect.OPENAI

    def test_gemini_generate(self) -> None:
        assert detect("/v1beta/models/gemini-2.5-flash:generateContent") is Dialect.GEMINI

    def test_gemini_stream(self) -> None:
        assert detect("/v1beta/models/gemini-2.5-flash:streamGenerateContent") is Dialect.GEMINI

    def test_unknown(self) -> None:
        assert detect("/v1/models") is Dialect.UNKNOWN


class TestGeminiModelFromPath:
    def test_extracts_model(self) -> None:
        assert gemini_model_from_path("/v1beta/models/gemini-2.5-flash:generateContent") == (
            "gemini-2.5-flash"
        )

    def test_extracts_model_stream(self) -> None:
        assert gemini_model_from_path("/v1beta/models/gemini-2.5-pro:streamGenerateContent") == (
            "gemini-2.5-pro"
        )

    def test_no_models_marker(self) -> None:
        assert gemini_model_from_path("/v1/messages") is None

    def test_no_method_suffix(self) -> None:
        assert gemini_model_from_path("/v1beta/models/gemini-2.5-flash") is None

    def test_empty_model(self) -> None:
        assert gemini_model_from_path("/v1beta/models/:generateContent") is None


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


def _gemini_body(
    *,
    contents: list[dict[str, object]],
    system: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {"contents": contents}
    if system is not None:
        body["systemInstruction"] = system
    if extra is not None:
        body.update(extra)
    return body


class TestParseGemini:
    def test_text_subset_with_system_and_model_from_path(self) -> None:
        body = _gemini_body(
            contents=[
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi there"}]},
            ],
            system={"parts": [{"text": "be terse"}]},
            extra={"tools": [{"functionDeclarations": []}]},
        )
        req = parse(Dialect.GEMINI, body, model="gemini-2.5-flash")
        assert req is not None
        assert req.dialect is Dialect.GEMINI
        assert req.model == "gemini-2.5-flash"  # from the path, not the body
        assert req.system == "be terse"
        assert req.messages[0].role is Role.USER
        assert req.messages[0].text == "hello"
        assert req.messages[1].role is Role.ASSISTANT  # gemini "model" → assistant
        assert req.messages[1].text == "hi there"
        assert req.tools_json is not None
        assert req.stream is False  # gemini streams by endpoint, not a body flag

    def test_system_optional(self) -> None:
        req = parse(
            Dialect.GEMINI, _gemini_body(contents=[{"role": "user", "parts": [{"text": "x"}]}])
        )
        assert req is not None
        assert req.system is None
        assert req.model is None  # no model passed

    def test_non_text_system_passes_through(self) -> None:
        body = _gemini_body(
            contents=[{"role": "user", "parts": [{"text": "x"}]}],
            system={"parts": [{"inlineData": {"mimeType": "image/png", "data": "..."}}]},
        )
        assert parse(Dialect.GEMINI, body) is None

    def test_multi_part_system_passes_through(self) -> None:
        body = _gemini_body(
            contents=[{"role": "user", "parts": [{"text": "x"}]}],
            system={"parts": [{"text": "a"}, {"text": "b"}]},
        )
        assert parse(Dialect.GEMINI, body) is None

    def test_missing_contents_passes_through(self) -> None:
        assert parse(Dialect.GEMINI, {"systemInstruction": {"parts": [{"text": "s"}]}}) is None

    def test_contents_not_a_list_passes_through(self) -> None:
        assert parse(Dialect.GEMINI, {"contents": "nope"}) is None

    def test_non_dict_content_item_passes_through(self) -> None:
        assert parse(Dialect.GEMINI, {"contents": ["nope"]}) is None

    def test_unknown_role_text_passes_through(self) -> None:
        body = _gemini_body(contents=[{"role": "system", "parts": [{"text": "x"}]}])
        assert parse(Dialect.GEMINI, body) is None  # only user/model are the text subset

    def test_multi_part_turn_passes_through_when_not_structured(self) -> None:
        body = _gemini_body(contents=[{"role": "user", "parts": [{"text": "a"}, {"text": "b"}]}])
        assert parse(Dialect.GEMINI, body) is None

    def test_non_text_part_passes_through_when_not_structured(self) -> None:
        body = _gemini_body(
            contents=[{"role": "user", "parts": [{"functionCall": {"name": "f", "args": {}}}]}]
        )
        assert parse(Dialect.GEMINI, body) is None

    def test_non_dict_system_passes_through(self) -> None:
        body = {"contents": [{"role": "user", "parts": [{"text": "x"}]}], "systemInstruction": "s"}
        assert parse(Dialect.GEMINI, body) is None  # systemInstruction must be a Content object

    def test_roleless_item_passes_through_when_not_structured(self) -> None:
        # A content item without the exact {role, parts} shape isn't the text subset.
        assert parse(Dialect.GEMINI, _gemini_body(contents=[{"parts": [{"text": "x"}]}])) is None

    def test_roleless_item_carried_when_structured(self) -> None:
        item = {"parts": [{"text": "x"}]}  # no role key → not text subset, but round-trippable
        req = parse(Dialect.GEMINI, _gemini_body(contents=[item]), structured=True)
        assert req is not None
        assert req.messages[0].raw == item
        assert req.messages[0].role is Role.USER  # default for the cache-key view


class TestParseGeminiStructured:
    def test_carries_tool_turn_verbatim(self) -> None:
        tool_turn = {
            "role": "model",
            "parts": [{"functionCall": {"name": "sh", "args": {"cmd": "ls"}}}],
        }
        body = _gemini_body(contents=[{"role": "user", "parts": [{"text": "run it"}]}, tool_turn])
        req = parse(Dialect.GEMINI, body, structured=True)
        assert req is not None
        assert req.messages[0].raw is None  # plain text turn → spans
        assert req.messages[1].raw == tool_turn  # structured turn → verbatim
        assert req.messages[1].role is Role.ASSISTANT

    def test_unknown_role_structured_defaults_user_but_carries_raw(self) -> None:
        weird = {"role": "function", "parts": [{"functionResponse": {"name": "f"}}]}
        req = parse(Dialect.GEMINI, _gemini_body(contents=[weird]), structured=True)
        assert req is not None
        assert req.messages[0].raw == weird
        assert req.messages[0].role is Role.USER  # cache-key view only; serialises from raw


class TestSerializeGemini:
    def test_text_round_trips(self) -> None:
        body = _gemini_body(
            contents=[
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi"}]},
            ],
            system={"parts": [{"text": "sys"}]},
            extra={"generationConfig": {"temperature": 0.2}},
        )
        req = parse(Dialect.GEMINI, body, model="gemini-2.5-flash")
        assert req is not None
        out = serialize(req, body)
        assert out["contents"] == [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
        ]
        assert out["systemInstruction"] == {"parts": [{"text": "sys"}]}
        assert out["generationConfig"] == {"temperature": 0.2}  # untouched passthrough field

    def test_no_system_leaves_no_system_instruction(self) -> None:
        body = _gemini_body(contents=[{"role": "user", "parts": [{"text": "x"}]}])
        req = parse(Dialect.GEMINI, body)
        assert req is not None
        out = serialize(req, body)
        assert "systemInstruction" not in out

    def test_structured_turn_round_trips_verbatim(self) -> None:
        body = _gemini_body(
            contents=[
                {"role": "user", "parts": [{"text": "run"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}}]},
            ]
        )
        req = parse(Dialect.GEMINI, body, structured=True)
        assert req is not None
        assert serialize(req, body) == body  # parse → serialize identity (modulo separators)
