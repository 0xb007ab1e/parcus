"""Tests for the JSONL dataset loader and dialect inference."""

from __future__ import annotations

from pathlib import Path

from parsimony.eval import load_jsonl
from parsimony.model import Dialect


def test_loads_and_infers_dialects(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"name":"a","path":"/v1/messages","body":{"messages":[]}}\n'
        "\n"  # blank line is ignored
        '{"dialect":"openai","body":{"messages":[]}}\n',
        encoding="utf-8",
    )
    samples = load_jsonl(path)
    assert len(samples) == 2
    assert samples[0].name == "a"
    assert samples[0].dialect is Dialect.ANTHROPIC
    assert samples[1].dialect is Dialect.OPENAI
    assert samples[1].name == "sample-3"  # default name from line number (blank line skipped)


def test_unknown_dialect_without_hint(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"body":{"messages":[]}}\n', encoding="utf-8")
    assert load_jsonl(path)[0].dialect is Dialect.UNKNOWN
