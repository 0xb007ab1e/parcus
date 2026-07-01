"""Tests for the heuristic + tiktoken tokenizers and the default-tokenizer factory."""

from __future__ import annotations

import pytest

from parcus.ports import TokenizerPort
from parcus.tokenize import HeuristicTokenizer, TiktokenTokenizer, _encoding, default_tokenizer

# Tests that require the real tiktoken vocab skip when it can't load (e.g. offline / no cached
# vocab) — the heuristic-fallback path is covered separately and does not need the network.
_needs_tiktoken = pytest.mark.skipif(
    _encoding(None) is None, reason="tiktoken vocab unavailable (offline / not cached)"
)


class TestHeuristicTokenizer:
    def test_empty_is_zero(self) -> None:
        assert HeuristicTokenizer().count("") == 0

    def test_short_text_is_at_least_one(self) -> None:
        assert HeuristicTokenizer().count("a") == 1

    def test_scales_with_length(self) -> None:
        tok = HeuristicTokenizer()
        assert tok.count("x" * 8) == 2
        assert tok.count("x" * 9) == 3  # rounds up

    def test_is_deterministic(self) -> None:
        tok = HeuristicTokenizer()
        assert tok.count("repeatable") == tok.count("repeatable")

    def test_satisfies_tokenizer_port(self) -> None:
        assert isinstance(default_tokenizer(), TokenizerPort)


class TestTiktokenTokenizer:
    def test_default_tokenizer_is_tiktoken(self) -> None:
        assert isinstance(default_tokenizer(), TiktokenTokenizer)

    def test_empty_is_zero(self) -> None:
        assert TiktokenTokenizer().count("") == 0

    @_needs_tiktoken
    def test_counts_are_positive_and_match_tiktoken(self) -> None:
        import tiktoken

        tok = TiktokenTokenizer()
        text = "The quick brown fox jumps over the lazy dog."
        expected = len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert tok.count(text) == expected > 0

    @_needs_tiktoken
    def test_never_expands_relative_to_chars(self) -> None:
        # A real BPE count is always <= character count for normal text.
        tok = TiktokenTokenizer()
        text = "please just go ahead and honestly summarize this really clearly"
        assert 0 < tok.count(text) <= len(text)

    def test_falls_back_to_heuristic_when_no_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the "encoding unavailable" path (e.g. offline, no cached vocab).
        monkeypatch.setattr("parcus.tokenize._encoding", lambda model: None)
        tok = TiktokenTokenizer()
        text = "some text of a known length here"
        assert tok.count(text) == HeuristicTokenizer().count(text)

    @_needs_tiktoken
    def test_special_token_text_does_not_raise(self) -> None:
        # Text that looks like a special token must be counted as ordinary text, not raise.
        assert TiktokenTokenizer().count("<|endoftext|> hello") > 0
