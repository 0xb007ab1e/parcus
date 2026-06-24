"""Tests for Track B context compaction."""

from __future__ import annotations

from parsimony.memory.compaction import compact_with_memory
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


class FakeMemory:
    """A MemoryPort stub returning fixed snippets and recording ingests."""

    def __init__(self, snippets: tuple[str, ...] = ()) -> None:
        self._snippets = snippets
        self.ingested: list[CanonicalRequest] = []

    def ingest(self, request: CanonicalRequest) -> None:
        self.ingested.append(request)

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        return self._snippets[:limit]


def _convo(n: int) -> CanonicalRequest:
    roles = (Role.USER, Role.ASSISTANT)
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=tuple(Message(role=roles[i % 2], spans=(Span(f"message {i}"),)) for i in range(n)),
    )


class TestCompaction:
    def test_unchanged_when_below_min_messages(self) -> None:
        req = _convo(6)
        assert compact_with_memory(req, FakeMemory(("x",)), min_messages=8) is req

    def test_unchanged_when_no_retrieval(self) -> None:
        req = _convo(12)
        assert compact_with_memory(req, FakeMemory(()), min_messages=8) is req

    def test_unchanged_when_nothing_older_to_drop(self) -> None:
        req = _convo(10)
        # keep_recent covers the whole convo -> window starts at 0 -> nothing to drop.
        assert compact_with_memory(req, FakeMemory(("x",)), keep_recent=10, min_messages=8) is req

    def test_window_steps_back_to_a_user_message(self) -> None:
        # keep_recent lands the window boundary on an assistant message; it must step back.
        out = compact_with_memory(_convo(12), FakeMemory(("fact",)), keep_recent=5, min_messages=8)
        assert out.messages[0].role is Role.USER

    def test_no_user_messages_leaves_request_unchanged(self) -> None:
        all_assistant = CanonicalRequest(
            dialect=Dialect.ANTHROPIC,
            model="m",
            messages=tuple(Message(role=Role.ASSISTANT, spans=(Span(f"a{i}"),)) for i in range(10)),
        )
        assert (
            compact_with_memory(all_assistant, FakeMemory(("x",)), min_messages=8) is all_assistant
        )

    def test_compacts_long_history(self) -> None:
        req = _convo(12)
        out = compact_with_memory(
            req, FakeMemory(("fact A", "fact B")), keep_recent=4, retrieve=2, min_messages=8
        )
        assert out is not req
        assert len(out.messages) < len(req.messages)  # older turns dropped
        assert out.messages[0].role is Role.USER  # window starts on a user message (valid)
        assert out.messages[0].text.startswith("Relevant earlier context")
        assert "fact A" in out.messages[0].text and "fact B" in out.messages[0].text
        # the most recent message is preserved verbatim
        assert out.messages[-1].text == req.messages[-1].text
