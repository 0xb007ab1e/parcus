# Design note — token-reduction frontier (post-roadmap survey)

> Status: **survey / design**, not yet implemented. Follows the completed token-reduction roadmap
> (issue #48, `docs/design/token-reduction-roadmap.md`). Surveys the current literature (arXiv) for
> **novel** token-reduction mechanisms and decides which fit parcus's tenets. Decisions taken
> become ADRs; anything adopted ships behind the existing correctness gates and carries the
> attributions recorded here.

## 1. The filter — parcus's hard constraints decide fit

A mechanism only fits if it clears **all** of these (from `CLAUDE.md` / the master ruleset):

- **Provider-blind proxy.** parcus sends discrete **text tokens** over public JSON APIs to a
  **frozen, third-party** serving model (Anthropic / OpenAI / Gemini / Groq). It cannot modify the
  model, inject embeddings, or reach its KV cache.
- **Requests-only.** It compresses *requests* and serves exact / near-duplicate cached *responses*;
  **response bytes are never modified**.
- **Local-only models.** Any learned compressor is a **local** model, lazy and opt-in — never an
  outbound call to optimize (that would be self-defeating).
- **Correctness is the gate; tokens are the objective.** No lossy transform ships without passing a
  measured no-regression gate (`parcus eval`). Lossy tiers are **off by default** and **fail open**.

This filter is decisive: it rules out an entire (heavily-cited) branch of the literature and points
to the one branch that's genuinely under-explored here.

## 2. Ruled out — soft-prompt / learned-token methods (record the exclusion)

These compress context into **continuous embedding vectors / memory slots / gist tokens** injected
into the serving model's representation space; decompression **requires a model trained or adapted
to consume them**. A provider-blind proxy to a hosted API **cannot** use any of them — the hosted
model has no such support and only accepts text. Documented here so they aren't re-proposed:

| Method | Attribution | Why it can't apply |
|---|---|---|
| Gist Tokens | Mu, Li, Goodman — NeurIPS 2023 — [arXiv:2304.08467](https://arxiv.org/abs/2304.08467) | Gisting is trained *into* the LM via a restricted attention mask; the same model must emit/attend the gist tokens. |
| ICAE (In-Context Autoencoder) | Ge, Hu, Wang, Chen, Wei — ICLR 2024 — [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) | Compresses to memory slots; the target LM itself is the fixed decoder conditioning on them. |
| 500xCompressor | Li, Su, Collier — ACL 2025 — [arXiv:2408.03094](https://arxiv.org/abs/2408.03094) | Feeds condensed K–V into the base model; needs embedding/KV injection no API exposes. |
| xRAG | Cheng, Wang, et al. — NeurIPS 2024 — [arXiv:2405.13792](https://arxiv.org/abs/2405.13792) | A modality bridge projects a retrieval embedding into the LM's input space. |

Only relevant if parcus ever self-hosts a model (out of scope by tenet today).

## 3. Already shipped — the filtering family

parcus's Tier-2 learned compressor **is** the LLMLingua v1 family (a small local LM prunes
low-perplexity tokens). Same *filtering* family: Selective Context (self-information) and
LLMLingua-2 (a token-classification encoder). LLMLingua-2 was **evaluated and cut** (issue #48,
`docs/validation/RESULTS.md`) — it mutated prose punctuation (`trade-off` → `trade - off`) and
never cleared the answer-preservation gate.

- LLMLingua — Jiang, Wu, Lin, Yang, Qiu — EMNLP 2023 — [arXiv:2310.05736](https://arxiv.org/abs/2310.05736)
- Selective Context — Li, Dong, Lin, Guerin — EMNLP 2023 — [arXiv:2310.06201](https://arxiv.org/abs/2310.06201)
- LLMLingua-2 — Pan, Wu, Jiang, et al. — ACL 2024 Findings — [arXiv:2403.12968](https://arxiv.org/abs/2403.12968)

**Key structural finding:** *every* method in this family deletes tokens and re-joins with its own
tokenizer, so **none preserve text verbatim**. That reframes where the new value is (§4.2).

## 4. Candidate mechanisms that fit (ranked)

### 4.1 Gemini explicit context caching — extend the `CacheStrategy` port  *(top pick)*

**No paper — an official provider mechanism. Zero correctness risk: it changes billing/transport,
never content.** parcus already models Anthropic explicit `cache_control` breakpoints (M1b) and a
preservation guard for automatic-prefix providers (OpenAI/DeepSeek). The one caching model it does
**not** have is Gemini's **explicit context-cache API**: register a stable prefix via
`cachedContents.create`, receive a handle, reference it on later calls, and manage TTL + a storage
cost.

- **What parcus would do:** a new `CacheModel` variant (`explicit_context_api`) + a Gemini dialect
  adapter/`CacheStrategy` that registers the large stable prefix, holds the handle (stateful),
  references it, and weighs reuse frequency vs. Gemini's per-hour storage charge (delete stale
  caches). Distinct from Gemini's newer *implicit* automatic caching.
- **Fit:** squarely on the "applicable to all providers" goal; safest possible win (no content
  transform → no gate needed); reuses the existing port abstraction.
- Docs: [Gemini caching](https://ai.google.dev/gemini-api/docs/caching) ·
  [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) ·
  [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) ·
  [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache)
- **Thresholds to respect (fail open below them):** OpenAI/Anthropic min cacheable ≈ 1,024 tokens;
  DeepSeek unit = 64 tokens. Sub-threshold or unstable prefix → forward unmodified.

### 4.2 Query-aware **extractive** context compression — a new lossy tier  *(the novel content lever)*

The genuinely under-explored direction, because it has the two properties parcus's current lossy
tiers lack: **verbatim** and **query-conditioned**.

- **Gap:** everything parcus has is either *task-agnostic token pruning* (LLMLingua v1 — mutates
  mid-sentence) or *whole-body elision* (drops entire stale `tool_result`s). Neither compresses a
  **retained large body down to the sentences relevant to the current instruction, kept verbatim**.
- **Why it fits better than the filtering family:** extractive methods select **whole verbatim
  sentences** — no mid-sentence mutation (the exact failure that killed LLMLingua-2), matching the
  conservative "change as little as we must" ethos. In an agentic harness the "query" is
  identifiable (the latest user turn / instruction), so query-conditioning is available.
- **Tenets honoured:** small **local** model (✓); lossy → **behind the answer-preservation gate,
  off by default, fail open** (same discipline as elision/dedup).
- **Niche vs. elision:** elision *drops whole stale* tool_results; this *compresses retained large*
  bodies to their relevant slices. Define the boundary so they compose, not overlap.
- **Cost / risk:** a local model dependency + a **new answer-preservation corpus** to validate;
  dropping a relevant span is an answer regression, so the gate must bite.

Attributions if adopted (extractive family + the question-aware pruner):

| Method | Attribution | Note |
|---|---|---|
| RECOMP (extractive + abstractive) | Xu, Shi, Choi — ICLR 2024 — [arXiv:2310.04408](https://arxiv.org/abs/2310.04408) | dual-encoder selects useful sentences; query-conditioned |
| Provence (DeBERTa sentence-pruning) | Chirkova, Formal, Nikoulina, Clinchant — ICLR 2025 — [arXiv:2501.16214](https://arxiv.org/abs/2501.16214) | sequence-labeling keep/drop mask; verbatim |
| EXIT (context-aware extractive) | Hwang, Cho, Jeong, Song, Han, Park — 2024 — [arXiv:2412.12559](https://arxiv.org/abs/2412.12559) | context-aware sentence keep/drop; verbatim |
| LongLLMLingua (question-aware) | Jiang, Wu, Luo, Li, Lin, Yang, Qiu — ACL 2024 — [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) | question-aware pruning (but deletes tokens → not verbatim) |

> Attribution caveat: EXIT / Provence author strings were drawn from search snippets + repos —
> confirm verbatim against the arXiv abstract pages before they land in code/NOTICE.

### 4.3 Disfavored — abstractive compression

CompAct (Yoon, Lee, Hwang, Jeong, Kang — EMNLP 2024 — [arXiv:2407.09014](https://arxiv.org/abs/2407.09014)),
RECOMP-abstractive, Nano-Capsulator — these **rewrite / paraphrase** context. Technically usable
(they touch only the request), but paraphrasing prompt content is a larger correctness bet than
extractive selection and cuts against the ethos. Not recommended.

## 5. Taxonomy anchor

**"Prompt Compression for Large Language Models: A Survey"** — Li, Liu, Su, Collier — 2024 —
[arXiv:2410.12388](https://arxiv.org/abs/2410.12388). Frames the field as **hard-prompt** (filtering
+ paraphrasing; human-readable, model-agnostic) vs **soft-prompt** (learned vectors;
model-specific). parcus lives entirely in *hard-prompt* because it is provider-blind — and within
that, it has *filtering* (Tier-2) but not *query-aware extractive*, which §4.2 targets. A companion
RAG-specific survey: [arXiv:2409.13385](https://arxiv.org/abs/2409.13385).

## 6. Recommendation & sequencing

1. **Gemini explicit context-cache adapter** (§4.1) — low-risk, high-leverage, extends the existing
   `CacheStrategy` port, no content change, no new gate. The clear next feature; an ADR + a Gemini
   dialect/strategy slice.
2. **Query-aware extractive compression tier** (§4.2) — the novel content mechanism; verbatim +
   query-conditioned; local model; **gated, off by default**. Bigger lift (new local dep + a
   validation corpus + a clear niche vs. elision). Prototype behind a new `parcus eval` gate before
   committing.

Everything else is already shipped (filtering), impossible here (soft-prompt), or disfavored
(abstractive). All adopted work holds the tenets: fail-open, correctness-gated, local-only,
requests-only, responses untouched.
