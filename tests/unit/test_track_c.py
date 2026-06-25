"""Tests for Track C summary compaction (pure function + engine wiring)."""

from __future__ import annotations

import json

from parcus.cache import CachePolicy, NullCache
from parcus.compress import LosslessCompressor
from parcus.memory.compaction import compact_by_summary
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor

OK = UpstreamResponse(200, (("content-type", "application/json"),), b"{}")


class FakeSummarizer:
    def __init__(self, text: str = "- summary line") -> None:
        self._text = text

    def summarize(self, texts: object, *, max_items: int = 5) -> str:
        return self._text


class FakeUpstream:
    def __init__(self) -> None:
        self.last: UpstreamRequest | None = None

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        self.last = request
        return OK


class FakeMemory:
    def ingest(self, request: CanonicalRequest) -> None:
        return

    def relevant(self, query: str, *, limit: int = 5) -> tuple[str, ...]:
        return ()


def _convo(n: int) -> CanonicalRequest:
    roles = (Role.USER, Role.ASSISTANT)
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=tuple(Message(role=roles[i % 2], spans=(Span(f"message {i}"),)) for i in range(n)),
    )


class TestCompactBySummary:
    def test_unchanged_when_short(self) -> None:
        req = _convo(6)
        assert compact_by_summary(req, FakeSummarizer(), min_messages=8) is req

    def test_unchanged_when_summary_empty(self) -> None:
        req = _convo(12)
        assert compact_by_summary(req, FakeSummarizer(""), min_messages=8) is req

    def test_unchanged_when_nothing_older(self) -> None:
        req = _convo(10)
        assert compact_by_summary(req, FakeSummarizer(), keep_recent=10, min_messages=8) is req

    def test_no_user_messages_unchanged(self) -> None:
        all_assistant = CanonicalRequest(
            dialect=Dialect.ANTHROPIC,
            model="m",
            messages=tuple(Message(role=Role.ASSISTANT, spans=(Span(f"a{i}"),)) for i in range(10)),
        )
        assert compact_by_summary(all_assistant, FakeSummarizer(), min_messages=8) is all_assistant

    def test_compacts_with_summary_block(self) -> None:
        req = _convo(12)
        out = compact_by_summary(req, FakeSummarizer("- key fact"), keep_recent=4, min_messages=8)
        assert out is not req
        assert len(out.messages) < len(req.messages)
        assert out.messages[0].role is Role.USER
        assert out.messages[0].text.startswith("Conversation summary so far")
        assert "- key fact" in out.messages[0].text
        assert out.messages[-1].text == req.messages[-1].text


def _long_anthropic(n: int = 12) -> bytes:
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message number {i} here"}
        for i in range(n)
    ]
    return json.dumps({"model": "claude-x", "messages": messages}).encode()


class TestEngineSummarizeMode:
    async def test_summarize_mode_compacts_and_labels(self) -> None:
        up = FakeUpstream()
        engine = ProxyEngine(
            upstream=up,
            compressor=LosslessCompressor(),
            cache=NullCache(),
            redactor=Redactor(),
            policy=CachePolicy(),
            config=EngineConfig(
                anthropic_upstream="https://a.test",
                openai_upstream="https://o.test",
                cache_enabled=False,
                memory_enabled=True,
                memory_summarize=True,
                memory_min_messages=8,
                memory_keep_recent=4,
            ),
            memory=FakeMemory(),
            summarizer=FakeSummarizer("- earlier decision"),
        )
        result = await engine.handle(
            "POST", "/v1/messages", [("x-api-key", "k")], _long_anthropic()
        )
        sent = json.loads(up.last.content)
        assert result.meta["memory"] == "summary"
        assert len(sent["messages"]) < 12
        assert "Conversation summary so far" in sent["messages"][0]["content"]
        assert "earlier decision" in sent["messages"][0]["content"]
        assert result.meta["tokens_after"] < result.meta["tokens_before"]
