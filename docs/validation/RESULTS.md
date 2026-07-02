# Real-provider validation — parcus in front of Groq

The first end-to-end validation of parcus against a **live provider** (Groq's OpenAI-compatible
Chat Completions endpoint, `llama-3.1-8b-instant`), pointing a client's base URL at `parcus serve`
exactly as an agentic harness would. It answers the question the synthetic/offline tests can't:
**does parcus cut real, provider-billed tokens, at what latency cost, without changing answers?**

> Method: the same prompt is sent (A) directly to Groq and (B) through parcus (Tier-0 lossless +
> Tier-1 aggressive filler). Savings are measured by **Groq's own `usage.prompt_tokens`** — ground
> truth, not parcus's estimate. 10 passes × 4 prompt sizes. No secrets in this doc (counts only).

## Ground-truth token savings (Groq's own tokenizer)

| size | prompt chars | raw tok | compressed tok | saved (ground truth) |
|---|--:|--:|--:|--:|
| small | 322 | 100 | 89 | 11 (11.0%) |
| medium | 869 | 211 | 178 | 33 (15.6%) |
| large | 1627 | 364 | 298 | 66 (18.1%) |
| xlarge | 4052 | 842 | 666 | 176 (20.9%) |
| **all** | | **1517** | **1231** | **286 (18.9%)** |

Savings are real and **scale with prompt size** (more mutable prose to remove) — the right
direction for agentic harnesses, whose per-turn payloads are large. Compression preserved the
answer: at `temperature=0`, the direct and via-parcus replies were **identical**.

## Latency (ms), 10 passes — direct-to-Groq vs via-parcus

| size | direct mean/med/max | via-parcus mean/med/max | overhead (mean) |
|---|--:|--:|--:|
| small | 198/155/326 | 171/157/276 | −27 |
| medium | 212/170/365 | 209/177/329 | −3 |
| large | 220/177/394 | 205/198/300 | −15 |
| xlarge | 269/264/346 | 304/300/400 | +34 |

**Overall: direct 225 ms, via-parcus 222 ms, overhead ≈ −3 ms.** parcus's added latency is in the
noise versus the provider round-trip; only the 4 KB xlarge prompt shows a real local cost (+34 ms
for tokenize + compress). parcus is effectively invisible latency-wise.

## Key finding: parcus's local token estimate was inflated (now fixed)

With the original **4-chars/token heuristic**, parcus's *reported* savings (`x-parcus-tokens-saved`)
overstated the Groq-billed reality by ~1.5–2.2× (e.g. xlarge: heuristic 30.8% vs real 20.9%). The
heuristic over-counted exactly the whitespace/filler that compression removes.

Comparing the raw prompts against Groq's real (Llama) tokenizer showed a **near-constant offset**:

| size | Groq (real) | tiktoken (cl100k) | Groq − tiktoken |
|---|--:|--:|--:|
| small | 100 | 65 | 35 |
| medium | 211 | 176 | 35 |
| large | 364 | 329 | 35 |
| xlarge | 842 | 807 | 35 |

The constant +35 is the provider's fixed chat-template overhead (role/BOS markers) that the
message text doesn't include. Because it's constant, a real BPE encoding's **saved-token delta is
accurate** even though the absolute count runs under the billed prompt. So parcus now uses
`tiktoken` by default (`TiktokenTokenizer`, heuristic fallback) — exact for OpenAI-family models,
a close, delta-accurate approximation elsewhere. The `x-parcus-tokens-*` headers are parcus's BPE
measurement of the request *text*; the `x-parcus-upstream-*` headers (captured from the response)
are the provider's **billed** truth.

## Streaming + cache (also validated live)

- **Streaming:** a `stream:true` request was compressed (request body) and the SSE response relayed
  back incrementally (23 chunks, ~177 ms to first, `data: [DONE]` terminator) — real streaming with
  compression works end-to-end.
- **Exact cache:** a repeated identical request was served from parcus in **~8 ms with no provider
  call** (`x-parcus-cache: hit`, byte-identical body) — the largest single saving.

## Limitations / honest notes

- **BPE token count is not provably monotonic at the compressor level:** compression always
  shrinks the *text* (chars) and near-always the tokens, but on rare inputs removing whitespace
  can re-merge into one extra BPE token. The engine's **never-cost-more guard** now makes token
  non-expansion a hard guarantee on the request path: if compression tokenizes to more tokens
  than its input, the original is forwarded — so parcus never bills more than not compressing.
- **Provider prompt-cache (PLAN Q3) not exercised on Groq:** Groq reports no `cache_read` tokens,
  so the "does injecting a breakpoint make the provider serve the prefix from cache?" question is
  Anthropic-specific. **Now answered:** the M1b section below records a live Anthropic run where
  injection took turn-2 `cache_read` from 0 to the full prefix (5627 tokens).
- Savings depend on how much *mutable prose* a prompt carries; tool schemas / structured history
  are immutable to the filler tiers (history compaction via the memory tier is the lever there).

_Measured 2026-07-01, Groq `llama-3.1-8b-instant`, 10 passes × {small, medium, large, xlarge}._

---

# M1b — provider prompt-cache injection (Anthropic)

parcus's **M1b** injects a provider cache breakpoint (Anthropic `cache_control`) onto a large,
re-sent **stable prefix** (system + tools + all but the final turn) so the provider serves it from
its prompt cache on the next turn — the dominant cost lever for tool/history-heavy harnesses, and
the Anthropic-specific answer to PLAN Q3 that Groq can't exercise. Unlike the filler tiers above,
this doesn't remove tokens — it changes how the provider **bills** them (Anthropic cache reads cost
~0.1× input). Ships **off by default** (`cache_inject`), enabled only after the live numbers below
confirm the win — a malformed `cache_control` would 400 a live request.

Measured with `qa/cache-inject/validate.py`: the same large-prefix request is sent through parcus
**twice** per condition (baseline `cache_inject` off vs on), with compression disabled and parcus's
own cache off, so the only variable is the injected breakpoint. The win is the **turn-2
`cache_read_input_tokens`** delta — the provider's own billed count, captured by parcus into
`upstream_usage` / `x-parcus-upstream-cache-read-tokens`.

## Harness verified offline (simulated upstream)

`validate.py --self-test` drives the **real engine + serializer** against a *fake* Anthropic that
caches a prefix only when it receives a `cache_control` marker — proving parcus injects only when
enabled and that the harness reads usage correctly. **These counts are simulated (fake upstream),
not provider ground truth:**

| condition | turn-1 (in / cache_w / cache_r) | turn-2 (in / cache_w / cache_r) |
|---|---|---|
| baseline (inject off) | 5000 / 0 / 0 | 5000 / 0 / 0 |
| inject on | 5000 / 5000 / 0 | 5000 / 0 / **5000** |

Turn-2 `cache_read`: **0 → 5000** — parcus injects the breakpoint (and only when enabled), and the
harness reads it back. This exercises the injection path end-to-end without a network call; it does
**not** substitute for the live provider run.

## Live provider validation — PASSED (Anthropic `claude-haiku-4-5`)

Run via `qa/cache-inject/validate.py` against the **real Anthropic API** — the counts below are
Anthropic's own billed usage (`x-parcus-upstream-*`), not an estimate:

| condition | turn-1 (in / cache_w / cache_r) | turn-2 (in / cache_w / cache_r) |
|---|---|---|
| baseline (inject off) | 5641 / 0 / 0 | 5641 / 0 / 0 |
| **inject on** | 14 / 5627 / 0 | 14 / 0 / **5627** |

Prefix ≈ 5041 parcus-tiktoken tokens ≈ **5627 Anthropic-billed**. **Turn-2 `cache_read`: 0 →
5627** — with injection on, Anthropic served the entire re-sent prefix from its prompt cache and
billed only the 14-token volatile tail; baseline paid full price on both turns. Meaning was
preserved (only the `cache_control` marker changed; request text untouched).

**Economics** (billed input-token-equivalents; Anthropic cache read ≈ 0.1×, write ≈ 1.25×):

- **baseline, 2 turns:** `2 × 5641` = **11,282**.
- **inject, 2 turns:** turn-1 `14 + 5627×1.25` (write) + turn-2 `14 + 5627×0.10` (read) ≈ **7,625**
  → **~32% fewer** across just two turns (the first turn pays the 1.25× write premium).
- **steady state, N re-sent turns:** baseline `N·P`; inject `1.25P + (N−1)·0.10P` → approaches
  **~90% off the re-sent prefix** as turns accumulate. This is the agentic-harness win: system
  prompt + tool schemas + history are re-sent every turn, and injection bills them at ~0.1× after
  the first turn.

**Cost to reproduce:** four calls (2 conditions × 2 turns), `max_tokens=16` — ~**$0.02** on
`claude-haiku-4-5` (default), ~$0.06 Sonnet-4.6, ~$0.10 Opus-4.8 (≈ `prefix_tokens × 3.35 ×
input_price`; see current [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)).

> `cache_inject` still ships **off by default** — flipping it on is a deliberate config decision
> now that the win is validated; the injection is fail-open and only acts on explicit-breakpoint
> providers above their cache minimum.

_Validated 2026-07-02, Anthropic `claude-haiku-4-5`, prefix ≈5.6k tokens, 2 turns × {baseline, inject}._
