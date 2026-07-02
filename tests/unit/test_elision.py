"""Tests for the tool-result elision compressor (M1d slice 3)."""

from __future__ import annotations

from parcus.compress.elision import ToolResultElider
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span

_STUB = "[tool result elided by parcus]"
_BIG = "x" * 500  # a payload comfortably larger than the stub → worth eliding


def _raw(role: Role, content: object) -> Message:
    return Message(role=role, spans=(), raw={"role": role.value, "content": content})


def _req(*messages: Message) -> CanonicalRequest:
    return CanonicalRequest(dialect=Dialect.ANTHROPIC, model="m", messages=tuple(messages))


def _tool_result(text: str, tool_use_id: str = "t1") -> dict[str, object]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


def test_elides_stale_keeps_recent_and_preserves_pairing() -> None:
    old = _raw(Role.USER, [{"type": "text", "text": "see:"}, _tool_result(_BIG)])
    mid = _raw(Role.ASSISTANT, [{"type": "text", "text": "ok"}])  # raw, but no tool_result
    recent = _raw(Role.USER, [_tool_result(_BIG, tool_use_id="t2")])
    question = Message(role=Role.USER, spans=(Span("now what?"),))  # plain-text, not raw
    req = _req(old, mid, recent, question)  # keep_recent=2 → cutoff=2 → indices 0,1 eligible

    out, stats = ToolResultElider(keep_recent=2).compress(req)

    old_content = out.messages[0].raw["content"]  # type: ignore[index]
    assert old_content[0] == {"type": "text", "text": "see:"}  # non-tool_result block verbatim
    assert old_content[1]["content"] == _STUB  # stale tool_result payload stubbed
    assert old_content[1]["tool_use_id"] == "t1"  # pairing preserved
    assert out.messages[1] is req.messages[1]  # mid raw msg has no tool_result → unchanged object
    assert out.messages[2].raw["content"][0]["content"] == _BIG  # recent → kept
    assert out.messages[3] is req.messages[3]  # plain-text turn untouched
    assert stats[0].step == "elision"
    assert stats[0].spans_touched == 1
    assert stats[0].tokens_after < stats[0].tokens_before  # a real reduction
    assert stats[0].ok is None


def test_small_payload_not_elided_never_cost_more() -> None:
    # A stale tool_result smaller than the stub would cost tokens to stub → skip it.
    req = _req(_raw(Role.USER, [_tool_result("ok")]))
    out, stats = ToolResultElider(keep_recent=0).compress(req)
    assert out is req
    assert stats == ()


def test_keep_recent_covers_all_leaves_request_unchanged() -> None:
    req = _req(_raw(Role.USER, [_tool_result(_BIG)]))  # keep_recent=4 → nothing eligible
    out, stats = ToolResultElider(keep_recent=4).compress(req)
    assert out is req
    assert stats == ()


def test_nonlist_content_preserved() -> None:
    req = _req(_raw(Role.ASSISTANT, "plain string content"))
    out, stats = ToolResultElider(keep_recent=0).compress(req)
    assert out is req  # content isn't a block list → nothing elided
    assert stats == ()


def test_empty_and_already_elided_payloads_are_skipped() -> None:
    msg = _raw(
        Role.USER,
        [
            {"type": "tool_result", "tool_use_id": "a", "content": _STUB},  # already stubbed
            {"type": "tool_result", "tool_use_id": "b", "content": None},  # empty
        ],
    )
    req = _req(msg)
    out, stats = ToolResultElider(keep_recent=0).compress(req)
    assert out is req  # already-stubbed + empty payloads → nothing elided
    assert stats == ()


def test_elides_large_block_list_payload() -> None:
    # tool_result content can itself be a block list; a large one is measured via its JSON length.
    big_blocks = [{"type": "text", "text": _BIG}]
    req = _req(
        _raw(Role.USER, [{"type": "tool_result", "tool_use_id": "t", "content": big_blocks}])
    )
    out, stats = ToolResultElider(keep_recent=0).compress(req)
    assert out.messages[0].raw["content"][0]["content"] == _STUB  # type: ignore[index]
    assert stats[0].spans_touched == 1


def test_fails_open_on_tokenizer_error() -> None:
    class _Boom:
        def count(self, text: str, model: str | None = None) -> int:
            raise RuntimeError("boom")

    req = _req(_raw(Role.USER, [_tool_result(_BIG)]))
    out, stats = ToolResultElider(keep_recent=0, tokenizer=_Boom()).compress(req)
    assert out is req  # fail open — original request, no stats
    assert stats == ()
