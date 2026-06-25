"""Tests for the Tier-1 filler guardrail invariant and its use in the eval runner."""

from __future__ import annotations

from parcus.compress import (
    DEFAULT_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LosslessCompressor,
)
from parcus.eval import evaluate, is_filler_equivalent
from parcus.eval.dataset import Sample
from parcus.eval.equivalence import filler_violations, removed_only_allowed
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span

ALLOWED = frozenset({"please", "just", "really"})


def _req(*spans: Span, system: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="m",
        messages=(Message(role=Role.USER, spans=spans),),
        system=system,
    )


class TestRemovedOnlyAllowed:
    def test_only_allowed_tokens_removed(self) -> None:
        assert removed_only_allowed("please fix this just now", "fix this now", ALLOWED) is True

    def test_removing_a_non_allowed_token_fails(self) -> None:
        assert removed_only_allowed("keep this", "this", ALLOWED) is False

    def test_added_token_fails(self) -> None:
        assert removed_only_allowed("a b", "a b c", ALLOWED) is False

    def test_reordering_fails(self) -> None:
        assert removed_only_allowed("a b", "b a", ALLOWED) is False

    def test_punctuation_is_normalised_for_membership(self) -> None:
        assert removed_only_allowed("fix this, please.", "fix this,", ALLOWED) is True


class TestFillerEquivalence:
    def test_filler_removal_is_equivalent(self) -> None:
        original = _req(Span("please fix this just now"))
        compressed = _req(Span("fix this now"))
        assert is_filler_equivalent(original, compressed, ALLOWED) is True

    def test_dropping_real_content_is_regression(self) -> None:
        original = _req(Span("fix this"))
        compressed = _req(Span("this"))
        assert is_filler_equivalent(original, compressed, ALLOWED) is False

    def test_immutable_span_must_be_preserved(self) -> None:
        original = _req(Span("```please```", mutable=False))
        compressed = _req(Span("``````", mutable=False))
        violations = filler_violations(original, compressed, ALLOWED)
        assert any("immutable" in v for v in violations)

    def test_system_filler_removal_is_equivalent(self) -> None:
        assert is_filler_equivalent(
            _req(Span("hi"), system="please be terse"),
            _req(Span("hi"), system="be terse"),
            ALLOWED,
        )


class TestFillerStructuralRegressions:
    def test_system_presence_change(self) -> None:
        assert not is_filler_equivalent(
            _req(Span("hi"), system="please be terse"),
            _req(Span("hi"), system=None),
            ALLOWED,
        )

    def test_system_content_change_beyond_fillers(self) -> None:
        assert not is_filler_equivalent(
            _req(Span("hi"), system="be terse"),
            _req(Span("hi"), system="be verbose"),
            ALLOWED,
        )

    def test_message_count_change(self) -> None:
        original = CanonicalRequest(
            dialect=Dialect.ANTHROPIC,
            model="m",
            messages=(
                Message(role=Role.USER, spans=(Span("a"),)),
                Message(role=Role.USER, spans=(Span("b"),)),
            ),
        )
        assert not is_filler_equivalent(original, _req(Span("a")), ALLOWED)

    def test_role_change(self) -> None:
        compressed = CanonicalRequest(
            dialect=Dialect.ANTHROPIC,
            model="m",
            messages=(Message(role=Role.ASSISTANT, spans=(Span("hi"),)),),
        )
        assert not is_filler_equivalent(_req(Span("hi")), compressed, ALLOWED)

    def test_span_structure_change(self) -> None:
        assert not is_filler_equivalent(_req(Span("a"), Span("b")), _req(Span("a b")), ALLOWED)


class TestFillerTierInRunner:
    def test_filler_chain_saves_tokens_and_passes_gate(self) -> None:
        sample = Sample(
            name="filled",
            dialect=Dialect.ANTHROPIC,
            body={
                "model": "m",
                "messages": [{"role": "user", "content": "please just fix the bug really now"}],
            },
        )
        report = evaluate(
            [sample],
            compressor=ChainCompressor([LosslessCompressor(), FillerCompressor()]),
            equivalence=lambda o, c: is_filler_equivalent(o, c, DEFAULT_FILLERS),
        )
        assert report.passed
        assert report.total_after < report.total_before  # fillers removed → real savings
