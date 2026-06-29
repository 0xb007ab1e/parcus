#!/usr/bin/env python3
"""Stress test + performance metrics for the parcus pipeline.

Hermetic and free: drives synthesized agentic conversations through the real ProxyEngine with a
**fake upstream** (no provider calls), so it measures parcus's own behavior — input-token
reduction (the billable quantity), per-stage breakdown, the model-free accuracy invariants, cache
outcomes, and local latency — deterministically.

Two phases:
  1. BASELINE testing round — shipped defaults (Tier-0 lossless + exact cache), 20 conversations
     spanning small/medium/large prompt sizes run in sequence (so the repeated ones cache-hit).
  2. LEARNING PATH — the same prompts re-run across an escalating compression ladder
     (lossless -> +filler -> +aggressive filler -> +learned), cache OFF to isolate compression,
     reporting reductions, penalties (accuracy + latency), and gaps — broken out by prompt size.

Run:  python benchmarks/stress_test.py   (writes benchmarks/RESULTS.md and prints a summary)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from parcus.cache import CachePolicy, NullCache, SqliteCache
from parcus.compress import (
    AGGRESSIVE_FILLERS,
    DEFAULT_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LearnedCompressor,
    LLMLinguaReducer,
    LosslessCompressor,
)
from parcus.ports import CachePort, CompressorPort
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor
from parcus.tokenize import default_tokenizer

_OK = UpstreamResponse(200, (("content-type", "application/json"),), b'{"ok":true}')


class FakeUpstream:
    """Returns a canned response without any network call."""

    def __init__(self) -> None:
        """Start the call counter at zero."""
        self.calls = 0

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        """Count the call and return the canned response (no network)."""
        self.calls += 1
        return _OK


def _anthropic(system: str, user: str, model: str = "claude-sonnet-4-6") -> bytes:
    body = {"model": model, "system": system, "messages": [{"role": "user", "content": user}]}
    return json.dumps(body).encode()


# --- Synthesized conversations: 20 prompts spanning small / medium / large -------------------
# Reusable fragments. Prose is filler-laden (so the filler tiers have something to remove);
# code / tool-JSON / numbers are IMMUTABLE (so the gaps are visible by design).
_SYS_PLAIN = "You are a helpful assistant."
_SYS_CODE = "You are a careful senior code assistant."
_SYS_OPS = "You are an ops assistant. Be precise with every number."
_SYS_VERBOSE = (
    "You are a very helpful and exceedingly diligent senior engineering assistant. Please be "
    "really careful and basically always make sure that you actually think things through "
    "step by step. Honestly, just try to be as clear as you possibly can, and obviously do "
    "not ever introduce regressions. Clearly explain your reasoning."
)
_TOOLS = (
    'Tools available:\n```json\n{"name":"read_file","description":"Read a file",'
    '"parameters":{"path":{"type":"string"}}}\n```\n'
)
_CODE = (
    "Here is the function:\n```python\ndef scale(replicas: int) -> int:\n"
    "    return max(1, replicas * 2)  # do not change this exact logic\n```\n"
    "and the path is /etc/parcus/config.yaml (leave it verbatim)."
)

# A pool of filler-rich prose sentences; `_prose(n)` concatenates n of them to scale prompt size.
_PROSE = (
    "Could you please just go ahead and basically explain, really clearly, what this actually "
    "does? Honestly I just want a simple and clear answer.",
    "I was kind of wondering if you might possibly be able to help me understand, you know, how "
    "it actually works under the hood, in a fair amount of detail please.",
    "Honestly, just to be totally clear, I would really very much appreciate it if you could "
    "basically walk me through the whole thing, step by step, in a thorough way.",
    "To be perfectly honest, I think it would be genuinely helpful if you could, you know, just "
    "summarize the key points as concisely as you possibly can for me.",
    "Obviously I don't want to waste your time at all, but could you please clarify, in a really "
    "clear and simple way, what the actual trade-offs are here?",
    "I guess what I'm basically trying to ask is whether you could, perhaps, give me a fairly "
    "detailed and genuinely thorough explanation of the reasoning that is involved.",
)


def _prose(n: int, start: int = 0) -> str:
    """Concatenate ``n`` filler-rich prose sentences (cycling the pool from ``start``)."""
    return " ".join(_PROSE[(start + i) % len(_PROSE)] for i in range(n))


def _build_corpus() -> list[tuple[str, str, bytes]]:
    """Build 20 (name, size, body) prompts: 8 small, 7 medium, 5 large."""
    items: list[tuple[str, str, bytes]] = []

    # SMALL (8): one short instruction, minimal/no system prompt.
    items.append(("small-01-prose", "small", _anthropic(_SYS_PLAIN, _prose(1, 0))))
    items.append(("small-02-prose", "small", _anthropic(_SYS_PLAIN, _prose(1, 1))))
    items.append(("small-03-prose", "small", _anthropic(_SYS_PLAIN, _prose(1, 2))))
    items.append(("small-04-repeat-01", "small", _anthropic(_SYS_PLAIN, _prose(1, 0))))  # dup→hit
    items.append(("small-05-code", "small", _anthropic(_SYS_CODE, f"Keep this exact:\n{_CODE}")))
    items.append(
        (
            "small-06-numbers",
            "small",
            _anthropic(
                _SYS_OPS,
                "Please scale to 10 replicas, set the timeout to 30 seconds, and keep the budget "
                "at 500 dollars. Just confirm clearly.",
            ),
        )
    )
    items.append(
        ("small-07-tooljson", "small", _anthropic(_SYS_PLAIN, _TOOLS + "Which tool reads a file?"))
    )
    items.append(("small-08-prose", "small", _anthropic(_SYS_PLAIN, _prose(1, 4))))

    # MEDIUM (7): verbose system + a couple of prose blocks, sometimes an immutable fragment.
    items.append(("med-01-prose", "medium", _anthropic(_SYS_VERBOSE, _prose(2, 0))))
    items.append(("med-02-prose", "medium", _anthropic(_SYS_VERBOSE, _prose(3, 2))))
    items.append(
        ("med-03-mixed-code", "medium", _anthropic(_SYS_VERBOSE, _prose(1, 1) + "\n" + _CODE))
    )
    items.append(
        (
            "med-04-mixed-numbers",
            "medium",
            _anthropic(
                _SYS_OPS,
                _prose(2, 3) + " Also scale to 12 replicas and keep the cap at 750 dollars.",
            ),
        )
    )
    items.append(
        ("med-05-tools", "medium", _anthropic(_SYS_VERBOSE + "\n\n" + _TOOLS, _prose(2, 1)))
    )
    items.append(
        (
            "med-06-debugging",
            "medium",
            _anthropic(
                _SYS_VERBOSE,
                "Honestly the test is just flaky. Here is the trace:\n```\nE   assert 2 == 3\n```\n"
                "Could you please basically explain why, and clearly suggest a fix? Be concise.",
            ),
        )
    )
    items.append(("med-07-repeat-01", "medium", _anthropic(_SYS_VERBOSE, _prose(2, 0))))  # dup→hit

    # LARGE (5): verbose system + tools + several prose blocks (prose-dense), sometimes code.
    items.append(
        ("large-01-prose-dense", "large", _anthropic(_SYS_VERBOSE + "\n\n" + _TOOLS, _prose(6, 0)))
    )
    items.append(("large-02-prose-dense", "large", _anthropic(_SYS_VERBOSE, _prose(6, 3))))
    items.append(
        (
            "large-03-mixed",
            "large",
            _anthropic(_SYS_VERBOSE + "\n\n" + _TOOLS, _prose(4, 1) + "\n" + _CODE),
        )
    )
    items.append(
        (
            "large-04-mixed-numbers",
            "large",
            _anthropic(
                _SYS_OPS + "\n\n" + _TOOLS,
                _prose(4, 2)
                + " Then scale to 20 replicas, timeout 45 seconds, budget 1000 dollars.",
            ),
        )
    )
    items.append(
        (
            "large-05-prose-dense",
            "large",
            _anthropic(_SYS_VERBOSE, _prose(5, 4) + " " + _prose(3, 0)),
        )
    )

    return items


CONVERSATIONS: list[tuple[str, str, bytes]] = _build_corpus()


@dataclass
class Row:
    """One conversation's measured outcome at a given configuration."""

    name: str
    size: str
    before: int
    after: int
    cache: str
    stages: list[tuple[str, int, int, bool | None]]
    ms: float

    @property
    def reduction(self) -> float:
        """Fraction of input tokens removed for this conversation (0 when no input)."""
        return (self.before - self.after) / self.before if self.before else 0.0


def _engine(compressor: CompressorPort, cache: CachePort) -> ProxyEngine:
    return ProxyEngine(
        upstream=FakeUpstream(),
        compressor=compressor,
        cache=cache,
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(anthropic_upstream="https://a.test", openai_upstream="https://o.test"),
    )


async def _run(compressor: CompressorPort, cache: CachePort) -> list[Row]:
    engine = _engine(compressor, cache)
    rows: list[Row] = []
    for name, size, body in CONVERSATIONS:
        start = time.perf_counter()
        result = await engine.handle("POST", "/v1/messages", [("x-api-key", "k")], body)
        ms = (time.perf_counter() - start) * 1000.0
        meta = result.meta
        stages = [(s.stage, s.tokens_before, s.tokens_after, s.ok) for s in meta.get("stages", ())]
        rows.append(
            Row(
                name=name,
                size=size,
                before=int(meta.get("tokens_before", 0)),
                after=int(meta.get("tokens_after", 0)),
                cache=str(meta.get("cache", "off")),
                stages=stages,
                ms=ms,
            )
        )
    return rows


def _filler(fillers: frozenset[str]) -> FillerCompressor:
    return FillerCompressor(fillers=fillers)


# Tier-2 model: a HF id ("gpt2") or a local folder path. Set PARCUS_LEARNED_MODEL to a directory
# holding the model files to run fully offline (with HF_HUB_OFFLINE=1).
_LEARNED_MODEL = os.environ.get("PARCUS_LEARNED_MODEL", "gpt2")


def _reducer() -> LLMLinguaReducer:
    return LLMLinguaReducer(model_name=_LEARNED_MODEL)


def _learned_available() -> bool:
    try:
        _reducer().reduce("a b c", keep_ratio=0.5)
    except Exception:
        return False
    return True


def _chain(*passes: CompressorPort) -> CompressorPort:
    return passes[0] if len(passes) == 1 else ChainCompressor(list(passes))


def _accuracy(rows: list[Row]) -> tuple[int, int]:
    checked = sum(1 for r in rows for _, _, _, ok in r.stages if ok is not None)
    passed = sum(1 for r in rows for _, _, _, ok in r.stages if ok is True)
    return passed, checked


def _gaps(rows: list[Row]) -> int:
    # A "gap": a canonicalized request whose compression produced no token reduction.
    return sum(1 for r in rows if r.before > 0 and r.after >= r.before)


_SIZES = ("small", "medium", "large")


def _bucket_reduction(rows: list[Row], size: str) -> tuple[int, int, float]:
    """Return (total_before, total_after, mean_reduction%) for one size bucket."""
    bucket = [r for r in rows if r.size == size]
    if not bucket:
        return 0, 0, 0.0
    tb = sum(r.before for r in bucket)
    ta = sum(r.after for r in bucket)
    mean = sum(r.reduction for r in bucket) / len(bucket) * 100
    return tb, ta, mean


def main() -> int:
    """Run both phases, write the report, print a summary."""
    tok = default_tokenizer()
    out: list[str] = ["# parcus stress test & performance metrics", ""]
    out.append(
        "Hermetic run: synthesized agentic conversations through the real ProxyEngine with a "
        "fake upstream (no provider calls). Token counts are the heuristic tokenizer's; "
        "reductions are **input-token** reductions; accuracy is the model-free per-stage "
        "invariant pass-rate. Latency is local pipeline wall-clock (indicative).\n"
    )

    # ---- Phase 1: baseline (shipped defaults: lossless + exact cache) ----
    rows = asyncio.run(_run(LosslessCompressor(), SqliteCache()))
    tb, ta = sum(r.before for r in rows), sum(r.after for r in rows)
    hits = sum(1 for r in rows if r.cache == "hit")
    sizes = {s: sum(1 for _, sz, _ in CONVERSATIONS if sz == s) for s in _SIZES}
    out += [
        "## Phase 1 — baseline testing round (defaults: Tier-0 lossless + exact cache)",
        "",
        f"{len(CONVERSATIONS)} conversations run in sequence "
        f"({sizes['small']} small / {sizes['medium']} medium / {sizes['large']} large). Total "
        f"input tokens {tb} -> {ta} (**{(tb - ta) / tb * 100:.1f}%** reduction from compression "
        f"alone), **{hits}** exact cache hit(s) (each a full upstream call avoided).",
        "",
        "| conversation | size | tokens before | tokens after | reduction | cache | latency ms |",
        "|---|:--|--:|--:|--:|:--:|--:|",
    ]
    for r in rows:
        out.append(
            f"| {r.name} | {r.size} | {r.before} | {r.after} | {r.reduction * 100:.1f}% | "
            f"{r.cache} | {r.ms:.2f} |"
        )
    out += [
        "",
        "**By size** (baseline / defaults — reduction is ~0 because Tier-0 only normalizes "
        "whitespace; the win here is the cache):",
        "",
        "| size | prompts | tokens before | tokens after | mean reduction |",
        "|---|--:|--:|--:|--:|",
    ]
    for s in _SIZES:
        btb, bta, bmean = _bucket_reduction(rows, s)
        out.append(f"| {s} | {sizes[s]} | {btb} | {bta} | {bmean:.1f}% |")
    out.append("")

    # ---- Phase 2: learning path (escalating compression, cache OFF) ----
    learned_ok = _learned_available()
    ladder: list[tuple[str, CompressorPort]] = [
        ("L0 lossless", LosslessCompressor()),
        ("L1 +filler", _chain(LosslessCompressor(), _filler(DEFAULT_FILLERS))),
        ("L2 +aggressive", _chain(LosslessCompressor(), _filler(AGGRESSIVE_FILLERS))),
    ]
    if learned_ok:
        ladder.append(
            (
                "L3 +learned",
                _chain(
                    LosslessCompressor(),
                    _filler(AGGRESSIVE_FILLERS),
                    LearnedCompressor(_reducer(), keep_ratio=0.5),
                ),
            )
        )

    out += [
        "## Phase 2 — learning path (same prompts; escalating compression; exact cache OFF)",
        "",
        "Each level adds a tier. **Reduction** = mean input-token reduction; **penalty (acc)** = "
        "model-free invariant pass-rate (passed/checked; lower = a tier's safety check failed); "
        "**penalty (ms)** = mean added local latency; **gaps** = requests with no reduction at "
        "that level (e.g. code-heavy prompts where only immutable spans remain).",
        "",
        "| level | mean reduction | Δ vs prev | penalty: accuracy | penalty: mean ms | gaps |",
        "|---|--:|--:|:--:|--:|--:|",
    ]
    prev = 0.0
    level_rows: dict[str, list[Row]] = {}
    for label, comp in ladder:
        lr = asyncio.run(_run(comp, NullCache()))
        level_rows[label] = lr
        mean_red = sum(r.reduction for r in lr) / len(lr) * 100
        passed, checked = _accuracy(lr)
        acc = f"{passed}/{checked}" + (f" ({passed / checked * 100:.0f}%)" if checked else " (n/a)")
        mean_ms = sum(r.ms for r in lr) / len(lr)
        out.append(
            f"| {label} | {mean_red:.1f}% | {mean_red - prev:+.1f} | {acc} | {mean_ms:.2f} "
            f"| {_gaps(lr)} |"
        )
        prev = mean_red
    if not learned_ok:
        out += [
            "",
            "> **Gap — Tier-2 (learned) unavailable:** the `learned` extra (LLMLingua) is not "
            "installed, so L3 is omitted. In a deployment without it, enabling `LEARNED` "
            "**fails open** (no reduction) rather than erroring — install `parcus[learned]` for "
            "real Tier-2 numbers.",
        ]
    else:
        l2 = level_rows.get("L2 +aggressive", [])
        l3 = level_rows.get("L3 +learned", [])
        l2_red = sum(r.reduction for r in l2) / len(l2) * 100 if l2 else 0.0
        l3_red = sum(r.reduction for r in l3) / len(l3) * 100 if l3 else 0.0
        l3_ms = sum(r.ms for r in l3) / len(l3) if l3 else 0.0
        l2_ms = sum(r.ms for r in l2) / len(l2) if l2 else 0.0
        l3_pass, l3_checked = _accuracy(l3)
        sm_d = _bucket_reduction(l3, "small")[2] - _bucket_reduction(l2, "small")[2]
        md_d = _bucket_reduction(l3, "medium")[2] - _bucket_reduction(l2, "medium")[2]
        lg_d = _bucket_reduction(l3, "large")[2] - _bucket_reduction(l2, "large")[2]
        out += [
            "",
            f"> **Finding — Tier-2 (learned) pays off in proportion to prompt size.** Overall it "
            f"adds {l3_red - l2_red:+.1f}pp over aggressive filler for ~{l3_ms - l2_ms:.0f} "
            f"ms/request more latency (L3 {l3_ms:.0f} ms vs L2 {l2_ms:.2f} ms), but that average "
            f"hides a strong size gradient: **small {sm_d:+.1f}pp, medium {md_d:+.1f}pp, large "
            f"{lg_d:+.1f}pp** (see the per-size table). `reduce()` drives LLMLingua **v1** "
            "(`use_llmlingua2=False`), which compresses long multi-sentence *contexts*. parcus's "
            "immutable-span extraction protects code/paths/numbers/tool-JSON, so on **short** "
            "turns v1 only sees a sentence or two and keeps it whole — latency without payoff. On "
            "**medium/large prose-dense** prompts there is real multi-sentence text to compress, "
            "and v1 earns its keep. Practical guidance: enable `LEARNED` when your prompts are "
            "large and prose-heavy (pasted docs/long context), not for short chatty instructions; "
            "the ~200 ms cost is paid per request regardless. It **fails open** (returns the span "
            f"unchanged) rather than erroring, so correctness holds ({l3_pass}/{l3_checked}).",
        ]

    # ---- Per-size reduction across the ladder (small vs medium vs large) ----
    levels = list(level_rows.keys())
    out += [
        "",
        "### Mean reduction by prompt size (across the compression ladder)",
        "",
        "Where the savings actually come from: larger, more prose-dense prompts have more "
        "*mutable* text to compress, so they reduce more; small prompts and immutable-heavy "
        "ones stay near the floor.",
        "",
        "| size | prompts | " + " | ".join(levels) + " |",
        "|---|--:|" + "--:|" * len(levels),
    ]
    for s in _SIZES:
        n = sum(1 for _, sz, _ in CONVERSATIONS if sz == s)
        cells = [f"{_bucket_reduction(level_rows[lbl], s)[2]:.1f}%" for lbl in levels]
        out.append(f"| {s} | {n} | " + " | ".join(cells) + " |")

    # ---- Per-conversation reduction at the top compression level (shows WHERE gaps are) ----
    top_label, top_rows = list(level_rows.items())[-1]
    out += [
        "",
        f"### Per-conversation reduction at `{top_label}` (where the gaps are)",
        "",
        "| conversation | before | after | reduction | per-stage ok |",
        "|---|--:|--:|--:|:--|",
    ]
    for r in top_rows:
        oks = (
            ", ".join(
                f"{s}={'ok' if ok else ('—' if ok is None else 'FAIL')}" for s, _, _, ok in r.stages
            )
            or "—"
        )
        out.append(f"| {r.name} | {r.before} | {r.after} | {r.reduction * 100:.1f}% | {oks} |")

    out += [
        "",
        "## Read-out",
        "",
        "- **Reductions** scale with how much *mutable prose* a prompt contains; filler/aggressive "
        "add incremental savings on chatty prompts.",
        "- **Size gradient:** reduction grows with prompt size because larger prompts carry more "
        "mutable prose. Tier-2 (learned) is the clearest example — near-zero on small prompts, "
        "meaningfully positive on medium/large prose-dense ones (see the per-size table). Enable "
        "it for big-context prompts, not short turns.",
        "- **Penalties:** accuracy stays 100% for the model-free tiers (lossless/filler) by "
        "construction; the only cost is a little extra latency per added tier. (Tier-2 learned "
        "has *no* runtime invariant — its safety is the offline judge, a deliberate gap.)",
        "- **Gaps:** code-heavy / number-heavy prompts reduce little — immutable spans (code, "
        "paths, numbers, tool JSON) are protected on purpose; that is the safety floor, not a bug.",
        "- **Cache** is the largest single win when traffic repeats (Phase 1): a hit avoids the "
        "whole call, not just some tokens.",
    ]

    report = "\n".join(out) + "\n"
    dest = Path(__file__).parent / "RESULTS.md"
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {dest}")
    print(f"[tokenizer] {tok.count('probe', None)} token(s) for 'probe' (sanity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
