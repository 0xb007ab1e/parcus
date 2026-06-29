# parcus stress test & performance metrics

Hermetic run: synthesized agentic conversations through the real ProxyEngine with a fake upstream (no provider calls). Token counts are the heuristic tokenizer's; reductions are **input-token** reductions; accuracy is the model-free per-stage invariant pass-rate. Latency is local pipeline wall-clock (indicative).

## Phase 1 — baseline testing round (defaults: Tier-0 lossless + exact cache)

20 conversations run in sequence (8 small / 7 medium / 5 large). Total input tokens 3069 -> 3069 (**0.0%** reduction from compression alone), **2** exact cache hit(s) (each a full upstream call avoided).

| conversation | size | tokens before | tokens after | reduction | cache | latency ms |
|---|:--|--:|--:|--:|:--:|--:|
| small-01-prose | small | 43 | 43 | 0.0% | miss | 0.28 |
| small-02-prose | small | 47 | 47 | 0.0% | miss | 0.18 |
| small-03-prose | small | 48 | 48 | 0.0% | miss | 0.14 |
| small-04-repeat-01 | small | 43 | 43 | 0.0% | hit | 0.12 |
| small-05-code | small | 63 | 63 | 0.0% | miss | 0.16 |
| small-06-numbers | small | 43 | 43 | 0.0% | miss | 0.15 |
| small-07-tooljson | small | 43 | 43 | 0.0% | miss | 0.14 |
| small-08-prose | small | 45 | 45 | 0.0% | miss | 0.13 |
| med-01-prose | medium | 156 | 156 | 0.0% | miss | 0.18 |
| med-02-prose | medium | 199 | 199 | 0.0% | miss | 0.20 |
| med-03-mixed-code | medium | 170 | 170 | 0.0% | miss | 0.20 |
| med-04-mixed-numbers | medium | 106 | 106 | 0.0% | miss | 0.17 |
| med-05-tools | medium | 192 | 192 | 0.0% | miss | 0.21 |
| med-06-debugging | medium | 120 | 120 | 0.0% | miss | 0.18 |
| med-07-repeat-01 | medium | 156 | 156 | 0.0% | hit | 0.17 |
| large-01-prose-dense | large | 346 | 346 | 0.0% | miss | 0.29 |
| large-02-prose-dense | large | 316 | 316 | 0.0% | miss | 0.26 |
| large-03-mixed | large | 318 | 318 | 0.0% | miss | 0.27 |
| large-04-mixed-numbers | large | 221 | 221 | 0.0% | miss | 0.22 |
| large-05-prose-dense | large | 394 | 394 | 0.0% | miss | 0.29 |

**By size** (baseline / defaults — reduction is ~0 because Tier-0 only normalizes whitespace; the win here is the cache):

| size | prompts | tokens before | tokens after | mean reduction |
|---|--:|--:|--:|--:|
| small | 8 | 375 | 375 | 0.0% |
| medium | 7 | 1099 | 1099 | 0.0% |
| large | 5 | 1595 | 1595 | 0.0% |

## Phase 2 — learning path (same prompts; escalating compression; exact cache OFF)

Each level adds a tier. **Reduction** = mean input-token reduction; **penalty (acc)** = model-free invariant pass-rate (passed/checked; lower = a tier's safety check failed); **penalty (ms)** = mean added local latency; **gaps** = requests with no reduction at that level (e.g. code-heavy prompts where only immutable spans remain).

| level | mean reduction | Δ vs prev | penalty: accuracy | penalty: mean ms | gaps |
|---|--:|--:|:--:|--:|--:|
| L0 lossless | 0.0% | +0.0 | 20/20 (100%) | 0.20 | 20 |
| L1 +filler | 13.4% | +13.4 | 40/40 (100%) | 0.27 | 2 |
| L2 +aggressive | 16.9% | +3.6 | 40/40 (100%) | 0.43 | 2 |
| L3 +learned | 21.0% | +4.0 | 40/40 (100%) | 197.45 | 2 |

> **Finding — Tier-2 (learned) pays off in proportion to prompt size.** Overall it adds +4.0pp over aggressive filler for ~197 ms/request more latency (L3 197 ms vs L2 0.43 ms), but that average hides a strong size gradient: **small +0.0pp, medium +7.5pp, large +5.7pp** (see the per-size table). `reduce()` drives LLMLingua **v1** (`use_llmlingua2=False`), which compresses long multi-sentence *contexts*. parcus's immutable-span extraction protects code/paths/numbers/tool-JSON, so on **short** turns v1 only sees a sentence or two and keeps it whole — latency without payoff. On **medium/large prose-dense** prompts there is real multi-sentence text to compress, and v1 earns its keep. Practical guidance: enable `LEARNED` when your prompts are large and prose-heavy (pasted docs/long context), not for short chatty instructions; the ~200 ms cost is paid per request regardless. It **fails open** (returns the span unchanged) rather than erroring, so correctness holds (40/40).

### Mean reduction by prompt size (across the compression ladder)

Where the savings actually come from: larger, more prose-dense prompts have more *mutable* text to compress, so they reduce more; small prompts and immutable-heavy ones stay near the floor.

| size | prompts | L0 lossless | L1 +filler | L2 +aggressive | L3 +learned |
|---|--:|--:|--:|--:|--:|
| small | 8 | 0.0% | 12.9% | 15.8% | 15.8% |
| medium | 7 | 0.0% | 14.2% | 18.2% | 25.7% |
| large | 5 | 0.0% | 12.9% | 17.0% | 22.7% |

### Per-conversation reduction at `L3 +learned` (where the gaps are)

| conversation | before | after | reduction | per-stage ok |
|---|--:|--:|--:|:--|
| small-01-prose | 43 | 28 | 34.9% | lossless=ok, filler=ok, learned=— |
| small-02-prose | 47 | 43 | 8.5% | lossless=ok, filler=ok, learned=— |
| small-03-prose | 48 | 37 | 22.9% | lossless=ok, filler=ok, learned=— |
| small-04-repeat-01 | 43 | 28 | 34.9% | lossless=ok, filler=ok, learned=— |
| small-05-code | 63 | 63 | 0.0% | lossless=ok, filler=ok, learned=— |
| small-06-numbers | 43 | 38 | 11.6% | lossless=ok, filler=ok, learned=— |
| small-07-tooljson | 43 | 43 | 0.0% | lossless=ok, filler=ok, learned=— |
| small-08-prose | 45 | 39 | 13.3% | lossless=ok, filler=ok, learned=— |
| med-01-prose | 156 | 119 | 23.7% | lossless=ok, filler=ok, learned=— |
| med-02-prose | 199 | 110 | 44.7% | lossless=ok, filler=ok, learned=— |
| med-03-mixed-code | 170 | 148 | 12.9% | lossless=ok, filler=ok, learned=— |
| med-04-mixed-numbers | 106 | 70 | 34.0% | lossless=ok, filler=ok, learned=— |
| med-05-tools | 192 | 158 | 17.7% | lossless=ok, filler=ok, learned=— |
| med-06-debugging | 120 | 92 | 23.3% | lossless=ok, filler=ok, learned=— |
| med-07-repeat-01 | 156 | 119 | 23.7% | lossless=ok, filler=ok, learned=— |
| large-01-prose-dense | 346 | 284 | 17.9% | lossless=ok, filler=ok, learned=— |
| large-02-prose-dense | 316 | 254 | 19.6% | lossless=ok, filler=ok, learned=— |
| large-03-mixed | 318 | 277 | 12.9% | lossless=ok, filler=ok, learned=— |
| large-04-mixed-numbers | 221 | 194 | 12.2% | lossless=ok, filler=ok, learned=— |
| large-05-prose-dense | 394 | 194 | 50.8% | lossless=ok, filler=ok, learned=— |

## Read-out

- **Reductions** scale with how much *mutable prose* a prompt contains; filler/aggressive add incremental savings on chatty prompts.
- **Size gradient:** reduction grows with prompt size because larger prompts carry more mutable prose. Tier-2 (learned) is the clearest example — near-zero on small prompts, meaningfully positive on medium/large prose-dense ones (see the per-size table). Enable it for big-context prompts, not short turns.
- **Penalties:** accuracy stays 100% for the model-free tiers (lossless/filler) by construction; the only cost is a little extra latency per added tier. (Tier-2 learned has *no* runtime invariant — its safety is the offline judge, a deliberate gap.)
- **Gaps:** code-heavy / number-heavy prompts reduce little — immutable spans (code, paths, numbers, tool JSON) are protected on purpose; that is the safety floor, not a bug.
- **Cache** is the largest single win when traffic repeats (Phase 1): a hit avoids the whole call, not just some tokens.
