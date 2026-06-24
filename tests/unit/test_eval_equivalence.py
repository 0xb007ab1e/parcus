"""Tests for the lossless no-regression invariant checker."""

from __future__ import annotations

from parsimony.eval.equivalence import (
    is_lossless_equivalent,
    lossless_violations,
    word_sequence_equal,
)
from parsimony.model import CanonicalRequest, Dialect, Message, Role, Span


def _req(*messages: Message, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC, model="m", messages=tuple(messages), system=system
    )


def _user(*spans: Span) -> Message:
    return Message(role=Role.USER, spans=spans)


class TestWordSequenceEqual:
    def test_ignores_whitespace(self) -> None:
        assert word_sequence_equal("a  b\n\nc", "a b c") is True

    def test_detects_content_difference(self) -> None:
        assert word_sequence_equal("a b", "a c") is False


class TestLosslessEquivalence:
    def test_whitespace_only_change_is_equivalent(self) -> None:
        original = _req(_user(Span("hello   \n\n\nworld")))
        compressed = _req(_user(Span("hello\n\nworld")))
        assert is_lossless_equivalent(original, compressed) is True

    def test_altered_immutable_span_is_regression(self) -> None:
        original = _req(_user(Span("```x=1```", mutable=False)))
        compressed = _req(_user(Span("```x=2```", mutable=False)))
        violations = lossless_violations(original, compressed)
        assert violations
        assert any("immutable" in v for v in violations)

    def test_changed_mutable_content_is_regression(self) -> None:
        original = _req(_user(Span("keep this")))
        compressed = _req(_user(Span("keep this extra")))
        assert is_lossless_equivalent(original, compressed) is False

    def test_system_whitespace_change_is_equivalent(self) -> None:
        assert is_lossless_equivalent(
            _req(_user(Span("hi")), system="be   \n\n\nterse"),
            _req(_user(Span("hi")), system="be\n\nterse"),
        )

    def test_system_content_change_is_regression(self) -> None:
        assert not is_lossless_equivalent(
            _req(_user(Span("hi")), system="be terse"),
            _req(_user(Span("hi")), system="be verbose"),
        )

    def test_system_presence_change_is_regression(self) -> None:
        assert not is_lossless_equivalent(
            _req(_user(Span("hi")), system="be terse"),
            _req(_user(Span("hi")), system=None),
        )

    def test_message_count_change_is_regression(self) -> None:
        assert not is_lossless_equivalent(
            _req(_user(Span("a")), _user(Span("b"))),
            _req(_user(Span("a"))),
        )

    def test_role_change_is_regression(self) -> None:
        original = _req(Message(role=Role.USER, spans=(Span("hi"),)))
        compressed = _req(Message(role=Role.ASSISTANT, spans=(Span("hi"),)))
        assert not is_lossless_equivalent(original, compressed)

    def test_span_structure_change_is_regression(self) -> None:
        original = _req(_user(Span("a"), Span("b")))
        compressed = _req(_user(Span("a b")))
        assert not is_lossless_equivalent(original, compressed)
