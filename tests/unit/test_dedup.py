"""Tests for the Tier-2 cross-turn dedup compressor."""

from __future__ import annotations

from parcus.compress.dedup import DedupCompressor
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span

_MARKER = "[identical block above — deduplicated by parcus]"
_BIG = "A" * 300  # comfortably over the default 200-char min


def _msg(role: Role, *spans: Span) -> Message:
    return Message(role=role, spans=spans)


def _req(*messages: Message) -> CanonicalRequest:
    return CanonicalRequest(dialect=Dialect.ANTHROPIC, model="m", messages=tuple(messages))


def test_dedups_repeated_block_keeping_first() -> None:
    req = _req(
        _msg(Role.USER, Span("here: "), Span(_BIG)), _msg(Role.USER, Span("again: "), Span(_BIG))
    )
    out, stats = DedupCompressor().compress(req)
    assert out.messages[0].spans[1].text == _BIG  # first occurrence kept
    assert out.messages[1].spans[1].text == _MARKER  # later copy referenced
    assert stats[0].step == "dedup"
    assert stats[0].spans_touched == 1
    assert stats[0].tokens_after < stats[0].tokens_before


def test_dedups_repeated_immutable_block() -> None:
    # A pasted file is an immutable (code) span; dedup is the one op allowed to reference it, since
    # the content is preserved in the first occurrence.
    code = "```\n" + "x = 1\n" * 60 + "```"
    req = _req(
        _msg(Role.USER, Span(code, mutable=False)), _msg(Role.USER, Span(code, mutable=False))
    )
    out, stats = DedupCompressor().compress(req)
    assert out.messages[1].spans[0].text == _MARKER
    assert stats[0].spans_touched == 1


def test_small_repeat_not_deduped() -> None:
    small = "s" * 50  # under the min → not worth deduping
    req = _req(_msg(Role.USER, Span(small)), _msg(Role.USER, Span(small)))
    out, stats = DedupCompressor().compress(req)
    assert out is req
    assert stats == ()


def test_distinct_large_blocks_kept() -> None:
    req = _req(_msg(Role.USER, Span("A" * 300)), _msg(Role.USER, Span("B" * 300)))
    out, stats = DedupCompressor().compress(req)
    assert out is req  # nothing repeated → unchanged
    assert stats == ()


def test_structured_message_skipped() -> None:
    raw_msg = Message(
        role=Role.USER,
        spans=(),
        raw={"role": "user", "content": [{"type": "text", "text": _BIG}]},
    )
    req = _req(_msg(Role.USER, Span(_BIG)), raw_msg)
    out, stats = DedupCompressor().compress(req)
    assert out is req  # raw content isn't scanned → the plain-text _BIG is a lone occurrence
    assert out.messages[1] is raw_msg
    assert stats == ()


def test_min_chars_clamped_above_marker() -> None:
    block = "z" * (len(_MARKER) - 1)  # shorter than the marker
    req = _req(_msg(Role.USER, Span(block)), _msg(Role.USER, Span(block)))
    out, stats = DedupCompressor(min_chars=1).compress(req)  # clamped to len(marker)+1
    assert (
        out is req
    )  # a block that isn't larger than the marker is never deduped (never-cost-more)
    assert stats == ()


def test_fails_open_on_tokenizer_error() -> None:
    class _Boom:
        def count(self, text: str, model: str | None = None) -> int:
            raise RuntimeError("boom")

    req = _req(_msg(Role.USER, Span(_BIG)), _msg(Role.USER, Span(_BIG)))
    out, stats = DedupCompressor(tokenizer=_Boom()).compress(req)
    assert out is req  # fail open — original request, no stats
    assert stats == ()
