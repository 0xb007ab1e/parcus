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
  Anthropic-specific. The capture plumbing (`x-parcus-upstream-cache-read-tokens`) is in place and
  verified; the M1b injection + its harness have landed (see the **M1b** section below), and the
  live Anthropic run needs a key.
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

## Live provider validation — PENDING (needs an Anthropic key)

Run: `ANTHROPIC_API_KEY=… .venv/bin/python qa/cache-inject/validate.py` (four cheap calls; default
`claude-haiku-4-5`, override with `--model`). Fill from its output:

| condition | model | prefix tok (parcus est.) | turn-2 cache_read (billed) |
|---|---|--:|--:|
| baseline (inject off) | _TBD_ | _TBD_ | _TBD (expect ~0)_ |
| inject on | _TBD_ | _TBD_ | _TBD (expect ≈ prefix)_ |

**Pass criterion:** inject-on turn-2 `cache_read` is ≫ baseline (≈ the stable-prefix token count),
confirming the provider served the re-sent prefix from its cache. On a pass, record the numbers
here and reconsider the `cache_inject` default. On no gain, check prefix size vs the model's cache
minimum (4096 for Opus/Haiku, 2048 for Sonnet-4.6/Fable) and that the prefix is byte-stable turn to
turn.

_Offline self-test 2026-07-02; live Anthropic numbers pending._
