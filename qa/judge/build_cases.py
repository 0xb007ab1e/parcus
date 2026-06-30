#!/usr/bin/env python3
"""Build lossy-tier judgement cases for the in-session ("active Claude") quality gate.

Option A of the validation plan: no API key. We compress a realistic, filler-rich agentic
corpus through parcus's **real** lossy tiers (Tier-1 default filler, Tier-2 aggressive filler,
Tier-2 learned when a local model is present) and emit each ``(original → compressed)`` pair to
``cases.json``. The active Claude session then judges, per case, whether the compressed prompt
would elicit the *same answer* — i.e. whether any instruction, constraint, number, or named
entity was lost — and writes ``verdicts.json``. ``score.py`` joins the two into ``RESULTS.md``.

This grades the **actual product output** (not a mock) with a real model (the session), giving
the off-by-default lossy tiers genuine quality data — the gap the synthetic ``KeywordRecallJudge``
can only approximate. Run fully locally; the learned tier uses the local LLMLingua model when
``PARCUS_LEARNED_MODEL`` / the ``learned`` extra are available, else it is skipped with a note.

Run:  python qa/judge/build_cases.py   (writes qa/judge/cases.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from parcus.compress import (
    AGGRESSIVE_FILLERS,
    DEFAULT_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LearnedCompressor,
    LLMLinguaReducer,
    LosslessCompressor,
)
from parcus.model import CanonicalRequest, Dialect, Message, Role, Span
from parcus.ports import CompressorPort
from parcus.tokenize import default_tokenizer

# Realistic agentic prompts: heavy in discourse filler (so the filler tiers act) AND carrying
# load-bearing content — hard constraints, numbers, named identifiers — so the judge can tell
# whether compression dropped anything that changes the answer. No secrets/PII (master §5).
CORPUS: tuple[tuple[str, str], ...] = (
    (
        "refactor-validate",
        "Could you please just go ahead and refactor the `parse_config` function so that it "
        "basically validates the input is a dict, and honestly make sure it raises ValueError "
        "on a missing 'host' key, really keeping the existing return type unchanged.",
    ),
    (
        "scale-deploy",
        "I was just wondering if you could simply scale the deployment to 5 replicas and set the "
        "memory limit to 512Mi, and please really don't touch the CPU request at all.",
    ),
    (
        "debug-flaky",
        "Honestly the test is just really flaky. Could you basically explain why `assert 2 == 3` "
        "would obviously fail, and clearly suggest a concrete fix? Please be concise.",
    ),
    (
        "summarize-constraint",
        "Please just summarize the meeting notes in exactly 3 bullet points, and honestly keep "
        "each bullet under 15 words — really don't add any preamble whatsoever.",
    ),
    (
        "api-instruction",
        "Could you kindly write a GET endpoint at /v1/health that simply returns 200 with a JSON "
        'body of {"status": "ok"}, and please make absolutely sure it never touches the '
        "database at all.",
    ),
    (
        "ordered-steps",
        "I'd really appreciate it if you could just walk me through, step by step, how to rotate "
        "the signing key: first generate the new key, then add it alongside the old one, and only "
        "then revoke the old key — please keep that exact order.",
    ),
    (
        "numbers-budget",
        "Please just confirm clearly that you'll set the timeout to 30 seconds, cap the budget at "
        "500 dollars, and retry at most 3 times with exponential backoff — honestly don't change "
        "any of those numbers.",
    ),
    (
        "negation-careful",
        "Could you basically review this PR, but please really make sure you do NOT approve it if "
        "there are any failing tests, and obviously do not merge it yourself under any "
        "circumstances.",
    ),
)


def _compress(prompt: str, compressor: CompressorPort) -> str:
    request = CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model="judge-eval",
        messages=(Message(role=Role.USER, spans=(Span(prompt, mutable=True),)),),
    )
    compressed, _ = compressor.compress(request)
    return compressed.messages[0].text


def _learned_chain() -> CompressorPort | None:
    """Build the lossless→aggressive→learned chain if a local LLMLingua model is available."""
    model = os.environ.get("PARCUS_LEARNED_MODEL", "gpt2")
    reducer = LLMLinguaReducer(model_name=model)
    try:
        reducer.reduce("probe", keep_ratio=0.5)
    except Exception:
        return None
    return ChainCompressor(
        [
            LosslessCompressor(),
            FillerCompressor(fillers=AGGRESSIVE_FILLERS),
            LearnedCompressor(reducer, keep_ratio=0.5),
        ]
    )


def main() -> int:
    """Compress the corpus through each lossy tier and write the judgement cases."""
    tok = default_tokenizer()
    tiers: dict[str, CompressorPort | None] = {
        "L1-filler": ChainCompressor(
            [LosslessCompressor(), FillerCompressor(fillers=DEFAULT_FILLERS)]
        ),
        "L2-aggressive": ChainCompressor(
            [LosslessCompressor(), FillerCompressor(fillers=AGGRESSIVE_FILLERS)]
        ),
        "L3-learned": _learned_chain(),
    }
    skipped = [name for name, comp in tiers.items() if comp is None]
    cases: list[dict[str, object]] = []
    for name, prompt in CORPUS:
        for tier, comp in tiers.items():
            if comp is None:
                continue
            compressed = _compress(prompt, comp)
            cases.append(
                {
                    "id": f"{name}::{tier}",
                    "tier": tier,
                    "name": name,
                    "original": prompt,
                    "compressed": compressed,
                    "tokens_before": tok.count(prompt, None),
                    "tokens_after": tok.count(compressed, None),
                    "identical": compressed == prompt,
                }
            )
    dest = Path(__file__).parent / "cases.json"
    dest.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    to_judge = [c for c in cases if not c["identical"]]
    print(f"[written] {dest} — {len(cases)} cases ({len(to_judge)} need judging)")
    if skipped:
        print(f"[note] tiers skipped (no local model): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
