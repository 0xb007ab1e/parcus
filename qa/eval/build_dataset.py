#!/usr/bin/env python3
"""Build the differential-eval dataset for the lossy-tier correctness gate.

The lossy compression tiers (aggressive filler, learned) change prompt *content*, so their
safety cannot be proven by the model-free invariants — it must be *judged* by running prompts
through a real model and checking the answer is preserved (master §4; OWASP-LLM lossy-output
risk). This script produces the input to that judgement, **using parcus locally only — no model
call, no network**: for each base task it emits the ORIGINAL prompt and the COMPRESSED prompt
(parcus's real Tier-0→Tier-1-aggressive chain), so promptfoo can later ask a model to answer the
compressed prompt and an LLM judge can score it against the original intent.

Output: ``qa/eval/dataset.yaml`` (a JSON document, which is valid YAML — avoids a YAML dep), a
list of promptfoo test rows ``{"vars": {"name", "original", "compressed"}}``.

Run:  python qa/eval/build_dataset.py   (then see qa/eval/README.md to run the graded eval)
"""

from __future__ import annotations

import json
from pathlib import Path

from parcus.compress import (
    AGGRESSIVE_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LosslessCompressor,
)
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span

# Base tasks: filler-rich prose with a clear, checkable intent (no secrets/PII — master §5). The
# prose is deliberately verbose so the lossy chain actually changes it; a faithful model should
# answer the compressed form exactly as it would the original.
_TASKS: tuple[tuple[str, str], ...] = (
    (
        "explain-proxy",
        "Could you please just go ahead and basically explain, really clearly and simply, what "
        "a token-thrift inference proxy actually does for an agentic coding tool?",
    ),
    (
        "caching-tradeoff",
        "I was honestly just wondering if you could very briefly summarize the single biggest "
        "trade-off involved in caching LLM responses, in a really clear and simple way.",
    ),
    (
        "lossless-vs-filler",
        "Could you kindly, and as concisely as you possibly can, explain the basic difference "
        "between lossless whitespace compression and filler-word removal?",
    ),
    (
        "proxy-steps",
        "Please just walk me through, really clearly, the basic steps a reverse proxy takes when "
        "it forwards a request upstream and then returns the response to the client.",
    ),
    (
        "when-not-to-cache",
        "Honestly, could you simply and very clearly list a couple of situations where you should "
        "definitely NOT cache an API response?",
    ),
    (
        "define-idempotent",
        "I'd really appreciate it if you could just briefly and very clearly define what it means "
        "for an HTTP request to be idempotent.",
    ),
)

# The exact lossy chain a deployment would enable for this gate: Tier-0 lossless then Tier-1
# aggressive filler (mirrors `parcus serve` with FILLER_AGGRESSIVE on).
_CHAIN = ChainCompressor([LosslessCompressor(), FillerCompressor(fillers=AGGRESSIVE_FILLERS)])


def compress(prose: str) -> str:
    """Return ``prose`` after parcus's Tier-0→Tier-1-aggressive compression (mutable spans only)."""
    request = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="claude-sonnet-4-6",
        messages=(Message(role=Role.USER, spans=(Span(prose, mutable=True),)),),
    )
    compressed, _ = _CHAIN.compress(request)
    return compressed.messages[0].text


def build() -> list[dict[str, dict[str, str]]]:
    """Build the promptfoo test rows (original + compressed prompt per base task)."""
    return [
        {"vars": {"name": name, "original": prose, "compressed": compress(prose)}}
        for name, prose in _TASKS
    ]


def main() -> int:
    """Write ``dataset.yaml`` next to this script and print a short summary."""
    rows = build()
    dest = Path(__file__).parent / "dataset.yaml"
    dest.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    changed = sum(1 for r in rows if r["vars"]["compressed"] != r["vars"]["original"])
    print(f"[written] {dest} — {len(rows)} tasks, {changed} compressed (differ from original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
