# ADR 0011 — Query-aware extractive compression tier (anchored on Provence)

- Status: Proposed
- Date: 2026-07-12
- Deciders: project author
- Related: ADR 0001 (ports & adapters, fail-open), ADR 0006 (Tier-2 learned compressor — offline
  gate, `ok=None`); `docs/design/token-reduction-frontier.md` §4.2; issue #82 (Candidate 2)

## Context

The frontier survey (`docs/design/token-reduction-frontier.md` §4.2, issue #82) identifies
**query-aware extractive compression** as the one genuinely under-explored lever that fits parcus's
tenets. Everything parcus has today is either:

- **task-agnostic token pruning** — Tier-2 learned (LLMLingua v1) removes low-perplexity tokens
  regardless of the request's intent, and re-joins with its own tokenizer (it does *not* preserve
  text verbatim); or
- **whole-body elision / dedup** — the structured passes drop an entire stale `tool_result` or
  replace a byte-identical repeated block with a reference.

Neither compresses a **retained large body down to the sentences relevant to the current
instruction, kept verbatim**. That is the gap: a body parcus decides to *keep* (a pasted doc, a
long file, retrieved context) is sent in full even when only a few of its sentences bear on what the
user is actually asking this turn.

Extractive selection is attractive precisely because it has the two properties parcus's current
lossy passes lack: it is **verbatim** (selects whole original sentences — no mid-sentence mutation,
the exact failure that killed the LLMLingua-2 evaluation, issue #48) and **query-conditioned** (in
an agentic harness the "query" is identifiable — the latest user turn / trailing instruction). It is
lossy, so like every lossy pass it must be gated by a measured no-regression bar and default off.

## Decision

1. **Add a query-aware extractive compression pass**, off by default, that selects the
   query-relevant **verbatim sentences** from large *retained* mutable bodies and drops the rest.
   The "query" is the request's trailing instruction / latest user turn (already immutable in the
   canonical model); the pass never alters it.

2. **Anchor the selector on Provence** (Chirkova, Formal, Nikoulina, Clinchant — ICLR 2025 —
   [arXiv:2501.16214](https://arxiv.org/abs/2501.16214)): a query-conditioned context pruner
   formulated as **sequence labeling** (keep/drop over the context), so its output is **verbatim
   retained text, not a paraphrase**, and it "dynamically detects the needed amount of pruning."
   That profile is exactly the fit — verbatim + query-conditioned + a single compact local model,
   usable out-of-the-box. (The specific base model / public checkpoint is **to confirm at
   implementation**; the ADR commits to the *method*, not a hard-coded checkpoint.)

3. **Local-only, lazy, opt-in — behind a port, like Tier-2.** The selector runs a **local** model
   only (never an outbound call to save tokens — that would be self-defeating). It sits behind an
   injected `SentenceSelector`-style port (mirroring Tier-2's `TokenReducer` seam): a fake drives
   the pass logic in CI; the Provence-backed adapter lives behind a new optional extra (lazy import;
   model loaded on first use; absent extra → the pass **fails open**, no extractive step). Only the
   pass's span/sentence handling is in the critical coverage gate; the model-dependent adapter body
   is `# pragma: no cover`.

4. **Lossy → no runtime invariant → `ok=None`, gated offline (same discipline as ADR 0006).**
   There is no cheap model-free proof that dropping a sentence preserved the answer, so the pass
   reports `ok=None` and its safety rests on an **offline answer-preservation gate** the operator
   runs before enabling — a new `parcus eval` mode over a **new adversarial corpus** where dropping
   a relevant sentence is a detectable answer regression. This gate + corpus are **model-free to
   build** and land **first** (CI-verifiable via a fake selector); the real selector is a gated
   follow-up prototype validated against the corpus before it may be enabled.

5. **Compose with elision/dedup; don't overlap (define the boundary).** Ordering in the lossy
   stage: whole-body **elision** (drop stale `tool_result`s) and **dedup** (collapse repeated
   blocks) run first; extractive then operates only on the **retained** large bodies that survive
   them — compressing what elision *keeps*, never re-touching what it *drops*. Guards:
   immutable spans (code, paths, URLs, quoted strings, tool JSON, numbers/IDs, the trailing
   instruction) are never eligible; the pass only engages on a body above a size threshold (small
   turns aren't worth the model cost and risk); and it selects **whole sentences** so the output is
   a byte-subsequence of the input's sentences.

6. **Fail open, always.** Selector error, absent extra, sub-threshold body, or an empty/degenerate
   selection → forward the body unchanged. As with every optimization, the worst case is "no
   extractive savings this turn"; the pass never raises and never mutates a sentence's bytes.

## Consequences

- (+) Unlocks compression of *retained* large bodies — the biggest untouched token pool for
  doc/context-heavy prompts — while staying **verbatim** (no mid-sentence mutation) and
  **query-conditioned**, matching parcus's conservative "change as little as we must" ethos.
- (+) Reuses the proven Tier-2 shape: port + fake for CI, local model behind an optional extra,
  offline eval gate, off-by-default — so the correctness discipline is already established.
- (+) The model-free **eval gate + adversarial corpus** ships first and is independently useful:
  it's the acceptance bar any future retained-body compressor must clear.
- (−) A new **local model dependency** (optional extra) and a **new corpus** to author and
  maintain — real cost, and the corpus quality *is* the safety of the feature.
- (−) Lossy content transformation whose safety rests on an operator running the offline gate
  (no per-request proof — `ok=None`). Dropping a sentence that *was* relevant is an answer
  regression the runtime can't catch; the design makes that explicit rather than implying a
  guarantee it can't provide.
- (−) Query identification is heuristic (trailing instruction / latest user turn). A mis-identified
  query degrades *selection quality* (caught by the gate), not verbatim-ness.

## Alternatives considered

- **RECOMP-extractive** ([arXiv:2310.04408](https://arxiv.org/abs/2310.04408)) and **EXIT**
  ([arXiv:2412.12559](https://arxiv.org/abs/2412.12559)) — also verbatim, query-/context-conditioned
  sentence selectors, and genuinely viable. Provence is anchored instead for its explicitly
  **robust, out-of-the-box** design and dynamic pruning-amount detection; the `SentenceSelector`
  port keeps the choice swappable, so a later benchmark could substitute one without touching the
  pass.
- **LongLLMLingua** ([arXiv:2310.06839](https://arxiv.org/abs/2310.06839)) — question-aware, but it
  **deletes tokens** (not whole sentences) and re-joins → not verbatim, the same mid-sentence
  failure mode as LLMLingua-2. Rejected on the verbatim requirement.
- **Abstractive / paraphrasing** (CompAct, Nano-Capsulator) — rewrites content; a larger
  correctness bet than selection and against the ethos. Rejected (frontier §4.3).
- **Soft-prompt / learned-token** (gist tokens, ICAE, xRAG) — need model-side embedding/KV
  injection a hosted API can't accept; impossible for a provider-blind proxy (frontier §2).
- **Ship the selector now (skip the gate-first sequencing)** — rejected: a lossy content transform
  with no runtime invariant must not land enabled or unvalidated. Gate + corpus first, selector as
  a gated prototype, mirrors ADR 0006 and this project's correctness-first sequencing.

## Attribution note

If the Provence adapter lands, add a NOTICE entry: *Provence — Chirkova, Formal, Nikoulina,
Clinchant (ICLR 2025), arXiv:2501.16214*. Confirm the model licence and the exact checkpoint before
distribution (`topic-license-compliance`, `std-supplychain`). Status stays **Proposed** until the
eval gate + corpus land and a prototype clears them.
