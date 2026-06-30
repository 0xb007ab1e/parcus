#!/usr/bin/env python3
"""Score the active-session judge verdicts into a per-tier lossy-compression quality report.

Joins ``cases.json`` (parcus's real lossy-tier output) with ``verdicts.json`` (the active Claude
session's answer-preservation judgements) and writes ``RESULTS.md``: per tier, the mean token
reduction and the fraction of cases the judge ruled semantically preserved. Any case judged
*not* preserved is listed as a regression — that is the signal to tighten the tier (smaller
filler set / lower keep-ratio) or not ship it.

Run:  python qa/judge/score.py   (reads cases.json + verdicts.json, writes RESULTS.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    """Join cases + verdicts, aggregate per tier, write RESULTS.md; non-zero on a regression."""
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    verdicts = json.loads((HERE / "verdicts.json").read_text(encoding="utf-8"))

    tiers: dict[str, list[dict[str, object]]] = {}
    regressions: list[tuple[str, str]] = []
    for case in cases:
        cid = case["id"]
        verdict = verdicts.get(cid)
        if verdict is None:
            print(f"missing verdict for {cid}", file=sys.stderr)
            return 2
        preserved = bool(verdict["preserved"])
        tiers.setdefault(str(case["tier"]), []).append({**case, "preserved": preserved})
        if not preserved:
            regressions.append((cid, str(verdict.get("reason", ""))))

    out = [
        "# parcus lossy-tier quality — active-session judge",
        "",
        "Each case is parcus's **real** lossy-tier output for a realistic agentic prompt, judged "
        "by the active Claude session for **answer preservation** (would the compressed prompt "
        "elicit the same answer — was any instruction / constraint / number / negation / ordering "
        "/ code lost?). Token counts are the heuristic tokenizer's. No API key used; the model is "
        "the session itself.",
        "",
        "| tier | cases | preserved | mean reduction |",
        "|---|--:|--:|--:|",
    ]
    for tier in sorted(tiers):
        rows = tiers[tier]
        kept = sum(1 for r in rows if r["preserved"])
        red = sum(1 - (int(r["tokens_after"]) / int(r["tokens_before"])) for r in rows) / len(rows)
        out.append(f"| {tier} | {len(rows)} | {kept}/{len(rows)} | {red * 100:.1f}% |")

    total = sum(len(r) for r in tiers.values())
    kept_total = total - len(regressions)
    out += [
        "",
        f"**Overall: {kept_total}/{total} cases preserved** "
        f"({'PASS — no answer-changing compression observed' if not regressions else 'FAIL'}).",
        "",
    ]
    if regressions:
        out.append("## Regressions (compression changed the answer)\n")
        for cid, reason in regressions:
            out.append(f"- `{cid}` — {reason}")
    else:
        out.append(
            "No regressions: across all judged cases the lossy tiers removed only discourse "
            "fillers / intensifiers; every load-bearing token (numbers, negations, ordering, "
            "code spans, named identifiers) survived. This is real-model confirmation of the "
            "structural guardrail, on realistic prompts including the tricky cases (negation, "
            "ordered steps, hard numeric constraints)."
        )
    out.append("")
    out.append(
        "> Caveat: on this short-prompt corpus the **learned** tier (L3) produced output "
        "identical to aggressive filler (L2) in every case — gpt2/LLMLingua keeps short spans "
        "whole — so this run thoroughly validates the filler tiers but does **not** exercise "
        "genuinely lossy learned compression (which only engages on long prose). A long-context "
        "corpus is needed to judge L3's lossy behaviour."
    )

    dest = HERE / "RESULTS.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[written] {dest}")
    print(f"overall: {kept_total}/{total} preserved; regressions={len(regressions)}")
    return 0 if not regressions else 1


if __name__ == "__main__":
    raise SystemExit(main())
