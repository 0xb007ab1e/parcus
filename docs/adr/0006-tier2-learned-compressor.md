# ADR 0006 — Tier-2 learned compressor: local, opt-in, offline-gated

- Status: Accepted
- Date: 2026-06-25
- Deciders: project author

## Context

The PLAN's tiered compression ends at **Tier-2: a learned compressor** (M4) — drop
low-information tokens with a model, beyond what Tier-0 (whitespace) and Tier-1 (allow-listed
fillers) remove. This is the last unbuilt core token-saving feature. It is fundamentally
different from the lower tiers: it is **lossy and semantic**, so there is **no model-free
invariant** that can prove the output preserves meaning. That changes how it must be guarded.

## Decision

1. **Off by default, opt-in.** `PARCUS_LEARNED=false`; `PARCUS_LEARNED_RATIO` (keep-ratio
   in `(0, 1]`, default 0.5). It is the **last** pass in the chain (lossless → filler → learned),
   so it only ever sees already-trimmed prose.
2. **Mutable spans only; fail open.** Like every tier it touches only mutable prose spans (code,
   paths, quotes, tool JSON, the trailing instruction are never altered) and returns the request
   unchanged on any error — a token optimizer must never break the harness.
3. **No runtime invariant → `ok=None`.** Tiers 0/1 self-check a model-free invariant per request
   (whitespace-only / allow-listed-fillers-only) and report `ok=True/False`. Tier-2 cannot —
   there is no cheap, local, model-free proof that a semantic rewrite preserved the answer. So it
   reports `ok=None` ("no runtime invariant; accuracy comes from the offline gate"), which the
   observability layer already understands.
4. **Correctness is gated offline, before enabling.** The gate is an **answer-preservation
   quality judge** (`parcus.eval.quality.LLMJudge`) run over the eval corpus with a **local**
   judge model — comparing answers to the original vs. compressed prompt. An operator runs this
   offline and only enables Tier-2 if it holds the no-regression bar. This is deliberately *not*
   a CI gate: it needs a model, and CI stays hermetic. CI instead covers the **mechanism** (span
   handling, fail-open, stats) to 100% via a fake reducer.
5. **Local model only, via a seam.** A `TokenReducer` port does the actual reduction.
   `LearnedCompressor` (tested with a fake) owns span handling; `LLMLinguaReducer` is the
   production adapter — **LLMLingua** behind the optional `learned` extra, lazily imported, the
   model loaded locally on first use, **never a network call** (a compressor that phoned a remote
   model would defeat the project's purpose). If the extra is absent, the reducer raises on first
   use and the compressor **fails open** (no Tier-2, proxy still works) rather than crashing.

## Consequences

- The full tiered pipeline from the PLAN is now in place; operators with a local model can trade
  measured quality for larger prose cuts, opt-in and gated.
- The honest limitation: Tier-2's safety rests on the operator running the offline judge gate —
  there is no per-request proof. The design makes that explicit (`ok=None`, docs, off-by-default)
  rather than implying a guarantee it can't provide.
- `parcus.compress.learned` is in the 100%-critical coverage gate; the model-dependent
  `LLMLinguaReducer.reduce` body is `# pragma: no cover` (exercised only with the extra + model).
- Follow-ups: a built-in `eval --learned` flow once a small local judge is bundled/cached; an
  LLMLingua-2 path; per-domain keep-ratio tuning.
