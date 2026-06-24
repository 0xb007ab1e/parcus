"""Tests for the chain compressor (sequencing multiple passes)."""

from __future__ import annotations

from parsimony.compress import ChainCompressor, FillerCompressor, LosslessCompressor
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(text: str, *, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=(Span(text),)),),
        system=system,
    )


class TestChainCompressor:
    def test_applies_passes_in_order_and_merges_stats(self) -> None:
        # lossless first (normalise whitespace), then filler (drop 'really').
        req = _req("really   fix this   \n\n\n\n", system="please   \n\n\n\nbe terse")
        out, stats = ChainCompressor([LosslessCompressor(), FillerCompressor()]).compress(req)
        assert [s.step for s in stats] == ["lossless", "filler"]
        # Compare by word sequence (boundary blank-line runs are collapsed, not stripped).
        assert out.messages[0].text.split() == ["fix", "this"]
        assert out.system.split() == ["be", "terse"]

    def test_empty_chain_returns_request_unchanged(self) -> None:
        req = _req("anything")
        out, stats = ChainCompressor([]).compress(req)
        assert out is req
        assert stats == ()

    def test_single_pass_chain_matches_that_pass(self) -> None:
        req = _req("please fix")
        out, stats = ChainCompressor([FillerCompressor()]).compress(req)
        assert out.messages[0].text.split() == ["fix"]
        assert len(stats) == 1
