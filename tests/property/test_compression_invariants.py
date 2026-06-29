"""Property-based tests for the compression invariants (Hypothesis).

Compression is a *meaning-preserving* transform, so we cannot enumerate inputs — instead we
assert the **invariants that must hold for every possible request**. Hypothesis synthesises
thousands of canonical requests (mixed mutable prose + immutable spans, varied roles, system
prompts, tool JSON) and checks, for each compressor, that:

* it **never expands** the token count;
* **immutable spans are reproduced byte-for-byte** and request structure is preserved;
* Tier-0 (lossless) differs from the input **only in whitespace** (``is_lossless_equivalent``);
* Tier-1 (filler) removes **only allow-listed tokens** (``is_filler_equivalent``), for both the
  default and aggressive sets, and for the lossless→filler chain;
* compression is **deterministic** and **idempotent**;
* each pass's model-free **self-check reports ``ok``** on valid input.

These complement the example-based tests in ``tests/unit/`` — same invariants, but exercised
against a far wider input space than hand-written cases reach.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from parcus.compress import (
    AGGRESSIVE_FILLERS,
    DEFAULT_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LosslessCompressor,
)
from parcus.invariants import is_filler_equivalent, is_lossless_equivalent
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
from parcus.tokenize import default_tokenizer

# --- Strategies: synthesise realistic canonical requests -------------------------------------

# Vocabulary mixes removable fillers with content words so both tiers have something to do.
_WORDS = st.sampled_from(
    [
        *sorted(AGGRESSIVE_FILLERS),
        "fix",
        "the",
        "bug",
        "deploy",
        "service",
        "config",
        "return",
        "scale",
        "cache",
        "review",
    ]
)
# Separators include multi-space and multi-newline runs so the lossless pass has whitespace
# to normalise (and the filler pass must preserve line structure).
_SEP = st.sampled_from([" ", "  ", "   ", "\n", "\n\n", "\n\n\n\n", " \n  ", "\t "])

# Prose (mutable): tokens joined by varied whitespace; may be empty.
_PROSE = st.lists(st.tuples(_WORDS, _SEP), max_size=12).map(
    lambda pairs: "".join(tok + sep for tok, sep in pairs)
)

# Immutable spans: code fences, paths, URLs, inline code, tool JSON — never altered, whatever
# their content (the engine honours the ``mutable=False`` flag directly for message spans).
_IMMUTABLE_TEXT = st.sampled_from(
    [
        "```python\ndef f(x):\n    return x * 2  # keep exact\n```",
        "/etc/parcus/config.yaml",
        "`inline_code`",
        '{"name": "read_file", "n": 42}',
        "https://example.com/a/b?q=1",
        "value is 10 replicas, 30 seconds, 500 dollars",
    ]
)

_SPAN = st.one_of(
    _PROSE.map(lambda t: Span(t, mutable=True)),
    _IMMUTABLE_TEXT.map(lambda t: Span(t, mutable=False)),
)
_SPANS = st.lists(_SPAN, min_size=1, max_size=5).map(tuple)
_MESSAGE = st.builds(
    Message,
    role=st.sampled_from(list(Role)),
    spans=_SPANS,
)
_REQUEST = st.builds(
    CanonicalRequest,
    dialect=st.just(Dialect.ANTHROPIC),
    model=st.one_of(st.none(), st.sampled_from(["claude-x", "gpt-4o", "m"])),
    messages=st.lists(_MESSAGE, min_size=1, max_size=4).map(tuple),
    system=st.one_of(st.none(), _PROSE),
    stream=st.booleans(),
    tools_json=st.one_of(st.none(), st.just('[{"name":"read_file"}]')),
)

_FILLER_SETS = st.sampled_from([DEFAULT_FILLERS, AGGRESSIVE_FILLERS])
_PROPERTY = settings(max_examples=200, deadline=None)


# --- Shared structural assertions ------------------------------------------------------------


def _assert_structure_preserved(before: CanonicalRequest, after: CanonicalRequest) -> None:
    """Every compressor must keep dialect/model/stream/tools and the message+span skeleton."""
    assert after.dialect == before.dialect
    assert after.model == before.model
    assert after.stream == before.stream
    assert after.tools_json == before.tools_json
    assert len(after.messages) == len(before.messages)
    for m_in, m_out in zip(before.messages, after.messages, strict=True):
        assert m_in.role == m_out.role
        assert len(m_in.spans) == len(m_out.spans)
        for s_in, s_out in zip(m_in.spans, m_out.spans, strict=True):
            assert s_in.mutable == s_out.mutable  # mutability flags never change


def _assert_immutable_spans_identical(before: CanonicalRequest, after: CanonicalRequest) -> None:
    """Immutable spans (code/paths/URLs/JSON) must be reproduced byte-for-byte."""
    for m_in, m_out in zip(before.messages, after.messages, strict=True):
        for s_in, s_out in zip(m_in.spans, m_out.spans, strict=True):
            if not s_in.mutable:
                assert s_out.text == s_in.text


# --- Tier-0: lossless --------------------------------------------------------------------------


class TestLosslessInvariants:
    """The lossless tier may differ from its input only by removing meaningless whitespace."""

    @given(req=_REQUEST)
    @_PROPERTY
    def test_preserves_meaning_and_structure(self, req: CanonicalRequest) -> None:
        out, stats = LosslessCompressor().compress(req)
        assert stats, "lossless must not fail open on a well-formed request"
        assert is_lossless_equivalent(req, out)  # whitespace-only difference
        assert stats[0].ok is True  # the live self-check agrees
        _assert_structure_preserved(req, out)
        _assert_immutable_spans_identical(req, out)
        assert stats[0].tokens_after <= stats[0].tokens_before  # never expands

    @given(req=_REQUEST)
    @_PROPERTY
    def test_deterministic_and_idempotent(self, req: CanonicalRequest) -> None:
        comp = LosslessCompressor()
        first, _ = comp.compress(req)
        again, _ = comp.compress(req)
        assert first == again  # deterministic: same input -> identical output
        twice, _ = comp.compress(first)
        assert twice.text == first.text  # idempotent: no whitespace left to remove


# --- Tier-1: filler (default + aggressive) ----------------------------------------------------


class TestFillerInvariants:
    """The filler tier may remove only allow-listed tokens — proven for any input + any set."""

    @given(req=_REQUEST, fillers=_FILLER_SETS)
    @_PROPERTY
    def test_removes_only_allowed_tokens(
        self, req: CanonicalRequest, fillers: frozenset[str]
    ) -> None:
        out, stats = FillerCompressor(fillers=fillers).compress(req)
        assert stats, "filler must not fail open on a well-formed request"
        assert is_filler_equivalent(req, out, fillers)  # only allow-listed tokens dropped
        assert stats[0].ok is True
        _assert_structure_preserved(req, out)
        _assert_immutable_spans_identical(req, out)
        assert stats[0].tokens_after <= stats[0].tokens_before

    @given(req=_REQUEST, fillers=_FILLER_SETS)
    @_PROPERTY
    def test_deterministic_and_idempotent(
        self, req: CanonicalRequest, fillers: frozenset[str]
    ) -> None:
        comp = FillerCompressor(fillers=fillers)
        first, _ = comp.compress(req)
        again, _ = comp.compress(req)
        assert first == again
        twice, _ = comp.compress(first)
        assert twice.text == first.text  # all fillers already gone -> no further change


# --- Cross-cutting + chain --------------------------------------------------------------------


class TestCompressionNeverExpands:
    """No tier may ever produce more tokens than it received (measured via the public tokenizer)."""

    @given(req=_REQUEST, fillers=_FILLER_SETS)
    @_PROPERTY
    def test_token_count_is_non_increasing(
        self, req: CanonicalRequest, fillers: frozenset[str]
    ) -> None:
        tok = default_tokenizer()
        before = tok.count(req.text, req.model)
        for comp in (LosslessCompressor(), FillerCompressor(fillers=fillers)):
            out, _ = comp.compress(req)
            assert tok.count(out.text, out.model) <= before


class TestChainPreservesFillerInvariant:
    """Lossless→filler composed still removes only allow-listed tokens and keeps immutables."""

    @given(req=_REQUEST, fillers=_FILLER_SETS)
    @_PROPERTY
    def test_chain_is_filler_equivalent(
        self, req: CanonicalRequest, fillers: frozenset[str]
    ) -> None:
        chain = ChainCompressor([LosslessCompressor(), FillerCompressor(fillers=fillers)])
        out, _ = chain.compress(req)
        assert is_filler_equivalent(req, out, fillers)
        _assert_structure_preserved(req, out)
        _assert_immutable_spans_identical(req, out)
