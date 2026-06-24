"""Tests for the heuristic tokenizer and the default-tokenizer factory."""

from __future__ import annotations

from parsimony.ports import TokenizerPort
from parsimony.tokenize import HeuristicTokenizer, default_tokenizer


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
