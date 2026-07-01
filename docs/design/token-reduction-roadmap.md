# Design note — token-reduction roadmap (post-0.2.0)

> Status: **design / research**, not a committed decision. Forward-looking mechanisms beyond
> the tiers already shipped. Grounds PLAN §7 research question **Q3** (provider-prompt-cache
> awareness) and extends PLAN §4 (compression) / §5 (graph memory). Decisions that land become
> ADRs; validated wins update `docs/validation/RESULTS.md`.

## 1. Where the tokens actually are

The Groq real-provider validation (`docs/validation/RESULTS.md`) is the anchor: Tier-0/1
filler compression netted **~11–21%**, scaling with prompt size, because the *compressible
surface* — mutable prose — is a thin slice of an agentic turn. The bulk of a Claude Code /
opencode turn is:

1. **The re-sent stable prefix** — system prompt + tool schemas, byte-identical every turn.
2. **The growing conversation history** — every prior turn, re-sent in full.
3. **Large `tool_result` bodies** — a file read or command output from many turns ago, still
   carried verbatim.

Filler removal barely touches these. The highest-value untapped mechanisms all target this
bulk — and the single biggest one does not remove tokens at all, it changes how the provider
*bills* them.

## 2. The provider axis (why "applicable to all providers" is the crux)

Token *removal* is provider-agnostic — a shorter request is fewer `prompt_tokens` on any
provider. **Prompt caching is where providers diverge the most**, and it is the highest-value
lever, so parcus needs a per-adapter capability descriptor rather than one hardcoded model.

| Provider | Caching model | Control knob | Reported as | Read discount |
|---|---|---|---|---|
| **Anthropic** | explicit breakpoint | `cache_control: {type:"ephemeral"}` (≤4 breakpoints; ≥4096 tok Opus / ≥2048 Sonnet-4.6/Fable) | `cache_creation_input_tokens`, `cache_read_input_tokens` | ~0.1× (write 1.25× @5m / 2× @1h) |
| **OpenAI** | automatic prefix | none (auto for prompts ≥1024 tok) | `usage.prompt_tokens_details.cached_tokens` | ~0.5× |
| **DeepSeek** | automatic prefix | none | cache hit/miss tokens | discounted |
| **Google Gemini** | explicit context-cache API | separate cache resource | cached-content tokens | discounted |
| **Groq** | **none** (as validated) | — | not reported | — |

parcus already captures both shapes into `ProviderUsage` (`x-parcus-upstream-cache-read-tokens`
/ `-cache-write-tokens`), so the observability is in place; what's missing is *acting* on it.

**Provider-capability abstraction.** Give each dialect adapter a `caching:` descriptor —
`none | automatic_prefix | explicit_breakpoint` — and let the engine adapt:

- **`explicit_breakpoint` (Anthropic):** parcus can both **preserve** and **inject**
  `cache_control`.
- **`automatic_prefix` (OpenAI/DeepSeek):** parcus can only **preserve** — injection is moot
  (the provider caches on its own); the job is to not perturb the stable prefix so the
  automatic cache keeps hitting.
- **`none` (Groq):** nothing to preserve; only request-shrinking applies.

**The universal invariant:** *cache-preservation* — request compression must never perturb the
provider-cacheable prefix — is safe and beneficial on every caching provider and a no-op on
non-caching ones. That is the "applicable to all providers" answer: parcus is
cache-neutral-or-better everywhere, and cache-*injecting* only where the provider exposes
explicit control. It composes with the existing fail-open tenet — if capability is unknown,
treat as `none` and shrink only.

## 3. Mechanisms, tiered by leverage

Column key — **Provider:** which providers it applies to. **Groq-testable:** can the existing
Groq key validate it against ground-truth `prompt_tokens` today?

### Tier 1 — highest leverage (target the re-sent bulk)

**M1a. Cache-preservation guard (defensive, universal).**
Caching is a strict prefix match: any byte change before the last breakpoint invalidates
everything after it (render order `tools → system → messages`). parcus's own request
compression, if it touches the stable prefix, can silently convert a ~0.1× cache-read into a
full-price cache-write — a **net loss**. The guard: detect the provider-cacheable prefix (an
Anthropic `cache_control` block, or the harness's stable system+tools span) and treat
everything ahead of it as immutable, compressing only the volatile tail.
*Provider:* all caching providers. *Groq-testable:* **no** (Groq doesn't cache) — logic is
unit-testable with fakes; ground-truth needs OpenAI (`cached_tokens` stays high after
compression) or Anthropic (`cache_read_input_tokens` preserved).

**M1b. Cache-injection (offensive, Anthropic-class only).**
When a request has a large stable prefix but no `cache_control`, parcus adds an ephemeral
breakpoint on the last stable block (respecting the token minimum + 4-breakpoint cap). A
proxy-level ~90% cut on the re-sent prefix, zero meaning change. Request-only, in scope.
*Provider:* `explicit_breakpoint` (Anthropic). *Groq-testable:* **no** — needs an Anthropic key
(verify `cache_read_input_tokens` jumps on the second identical request).

**M1c. History compaction on the streaming path.**
Track B (retrieval) and Track C (rolling summary) already exist but are off-by-default and
**bypassed on streams** — which is where Claude Code lives. Wire them into the stream path
behind the existing recall gate, fail-open.
*Provider:* all. *Groq-testable:* **yes** — savings appear in Groq `prompt_tokens`.

**M1d. Tool-result elision in history (new).**
Old `tool_result` blocks are re-sent verbatim but rarely needed in full; in later turns replace
a large stale result with a stub/summary + re-expand pointer. Lossy → rides the
answer-preservation gate. Provider-agnostic analog of Anthropic's server-side
`clear_tool_uses_20250919` context editing.
*Provider:* all. *Groq-testable:* **yes**.

**M1e. Tool-schema minification.**
Extend Tier-0 lossless from whitespace to structured-JSON canonicalization of the tools block
(dozens of verbose schemas, re-sent every turn). Lossless.
*Provider:* all. *Groq-testable:* **yes** (Groq supports tools; savings in `prompt_tokens`).

### Tier 2 — moderate

- **Cross-request/turn dedup** of repeated context (same file pasted twice) → content-address
  and collapse. *All providers; Groq-testable: yes.*
- **LLMLingua-2 upgrade** to the Tier-2 learned compressor (harder prose compression at equal
  quality; still local, offline-gated). *All; Groq-testable: yes.*
- **Semantic-cache reach** — widen the precision-gated near-duplicate hit rate (a hit skips the
  provider call entirely; validated ~8 ms). *All; Groq-testable: yes.*

### Tier 3 — out of scope by tenet (named so it's a decision, not an omission)

- **Output-token controls** (`max_tokens` caps, terse-output nudges, effort routing). Output is
  the pricier side, but this modifies request intent / response — collides with "never modify
  responses / never change results." Opt-in only, if ever.
- **Model routing / cascade to a cheaper model.** Biggest cost lever in the field; **explicitly
  excluded** by PLAN §2 ("not a router to cheaper models").
- **Batch API.** N/A for an interactive proxy.

## 4. Testability with the existing Groq key

**Yes — validatable now against Groq ground-truth `prompt_tokens`:** every request-shrinking
mechanism — M1c (history compaction), M1d (tool-result elision), M1e (tool-schema minify), and
all of Tier 2 (dedup, LLMLingua-2, semantic cache). These are the same class the current
RESULTS.md matrix already exercises, so the harness extends directly.

**No — not validatable on Groq:** the prompt-cache lever (M1a preservation, M1b injection),
because Groq neither caches prefixes nor reports cache tokens. This is not a parcus limitation —
it's the provider axis from §2. To validate:

- **A free-tier OpenAI key** exercises **M1a preservation** cheaply: OpenAI auto-caches prompts
  ≥1024 tokens and reports `cached_tokens`; confirm `cached_tokens` stays high after parcus
  compresses (i.e. we didn't bust the automatic cache).
- **An Anthropic key** exercises **M1b injection**: send a large-prefix request twice, confirm
  `cache_read_input_tokens` on the second is ~0 without parcus and large with parcus injecting
  `cache_control`.

Unit tests with fake upstreams cover the *logic* of M1a/M1b regardless of provider key.

## 5. Recommended sequencing

The ranking inverts the intuition that "better compression" is next: **M1a/M1b (prompt-cache)
outweigh every filler improvement combined**, and the capture plumbing already exists — the gap
RESULTS.md flags as untested (Q3).

1. **M1a cache-preservation guard** — universal, purely protective, unit-testable now; validate
   on a free OpenAI key.
2. **M1e tool-schema minify + M1c history-on-stream** — provider-agnostic, Groq-validatable
   immediately, wire up existing machinery.
3. **M1b cache-injection** — Anthropic key; the single largest per-turn win.
4. **M1d tool-result elision** — lossy, gated; largest removable-history chunk.
5. **Tier 2** — dedup, LLMLingua-2, semantic-cache reach; second-order.

Every step keeps the tenets: fail-open, correctness-gated, local-only models, requests-only
(responses never modified), and — new here — **cache-neutral-or-better on every provider**.
