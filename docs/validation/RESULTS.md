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

- **BPE token count is not provably monotonic:** compression is guaranteed to never expand the
  *text* (chars), and near-always reduces tokens, but on rare inputs removing whitespace can
  re-merge into one extra BPE token. A guard that forwards the original when compression doesn't
  reduce tokens would make token non-expansion a hard guarantee (candidate follow-up).
- **Provider prompt-cache (PLAN Q3) untested here:** Groq reports no `cache_read` tokens, so the
  "did compression bust the provider's prompt cache?" question is Anthropic-specific. The capture
  plumbing (`x-parcus-upstream-cache-read-tokens`) is in place and verified; it needs an Anthropic
  key to exercise.
- Savings depend on how much *mutable prose* a prompt carries; tool schemas / structured history
  are immutable to the filler tiers (history compaction via the memory tier is the lever there).

_Measured 2026-07-01, Groq `llama-3.1-8b-instant`, 10 passes × {small, medium, large, xlarge}._
