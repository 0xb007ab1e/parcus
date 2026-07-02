# Design note — structured-content parser (M1d foundation)

> Status: **plan / design**, not yet implemented. Scopes the expansion of parcus's canonical
> parser beyond the text-only subset — the PLAN's stated "top M2 task" and the prerequisite for
> **M1d (tool-result elision)**. Grounds issue #48. Decisions taken become ADRs; slices ship
> behind the fidelity gate below.

## 1. Why this is the top remaining item

parcus canonicalizes a request into `CanonicalRequest` and only optimizes what it can
canonicalize. Today the parser (`proxy/dialects.py`) accepts a **deliberately conservative
text-only subset**: Anthropic messages whose `content` is a string, OpenAI messages that are
exactly `{role, content: str}`. **Anything else returns `None` → the request passes through
unmodified (fail-open).**

For an agentic harness like Claude Code that is the *common* case, not the exception: tool-using
turns carry `content` as a **list of blocks** (`tool_use`, `tool_result`, `text`, `image`), and
OpenAI turns carry `tool_calls` / a `tool` role. So today parcus **optimizes nothing** on
tool-using traffic — no Tier-0/1 compression, no cache injection (M1b), no history compaction
(M1c). Expanding the parser is what makes every existing lever apply to real agentic traffic, and
it is the prerequisite for the one new lever left:

**M1d — tool-result elision:** old `tool_result` bodies (a file read, a command dump from many
turns ago) are re-sent verbatim every turn and dominate history size. Once represented, elision
replaces a *stale* tool_result body with a stub/summary + reference, keeping recent ones intact —
the provider-agnostic analog of Anthropic's server-side `clear_tool_uses` context editing.

## 2. The non-negotiable: byte-identical round-trip fidelity

The parser's expansion is **correctness-critical**: a mis-serialized `tool_use`/`tool_result`
(dropped id, reordered field, corrupted image payload) would **break the harness or change the
result** — the exact thing the fail-open tenet forbids. So the foundational property, ahead of any
optimization, is:

> **For any request we canonicalize but do not modify, `serialize(parse(body)) == body`
> byte-for-byte** (modulo the compact-JSON separators M1e already applies).

Corollary — **fail open on anything unmodeled:** if a body contains a block type/shape the parser
doesn't fully round-trip, it returns `None` and the request passes through untouched. We *widen*
the subset only as far as we can prove round-trip fidelity; everything else stays passthrough.
This keeps the blast radius of the expansion at zero for un-handled shapes.

## 3. Model changes

`Message.content` is currently reconstructed from `spans: tuple[Span, ...]` (text runs). Extend to
represent structured content without losing the current string form:

- Introduce a **content-block** representation: a message's content is either the existing
  string-spans form (unchanged) or an ordered list of typed blocks. Candidate block kinds:
  - `text` — **mutable** (compressible; classified into spans as today).
  - `tool_use` / OpenAI `tool_calls` — **immutable** (id + name + input JSON reproduced verbatim).
  - `tool_result` / OpenAI `tool` role — **immutable by default, elidable** (the M1d target):
    carries a `tool_use_id` + a payload; the payload is what elision may summarize/stub.
  - `image` / `document` — **immutable** (base64/URL reproduced verbatim; never a compression or
    elision target).
  - anything else → the message is not canonicalized (fail open).
- Keep `mutable`/immutable classification per block (mirrors the existing `Span.mutable`), so the
  compressor and injector touch only what's safe.
- Preserve every field needed for exact re-serialization (ids, `cache_control` already present,
  provider-specific keys) — round-trip fidelity (§2) drives the shape, not convenience.

## 4. Dialect specifics

- **Anthropic:** `content` as a list of blocks (`{"type": "text"|"tool_use"|"tool_result"|
  "image", ...}`); `tool_result.content` may itself be a string or a block list. Preserve
  `id`/`tool_use_id`, `is_error`, `cache_control`.
- **OpenAI:** tool calls live on the assistant message as `tool_calls: [...]`, and results come as
  separate `{role: "tool", tool_call_id, content}` messages (not inline blocks). The canonical
  model must map both dialects onto one block representation and serialize each back to its own
  shape — the anti-corruption boundary that keeps the core dialect-agnostic
  (`topic-architecture-patterns`).
- The `mcp_servers` / server-tool shapes and images must round-trip or trigger passthrough.

## 5. M1d elision mechanics (built on §3–4)

Once tool_result blocks are represented and round-trip-safe:

- Elide only **stale** results — beyond a `keep_recent` window (recent tool outputs are usually
  still load-bearing), mirroring the memory-compaction `keep_recent` knob.
- Replace the payload with a compact stub (`[tool result elided — N tokens; ask to re-run]`) or a
  short extractive summary; keep the `tool_use_id` and structure intact.
- **Lossy → answer-preservation gate:** ships behind the same eval gate as the other lossy tiers
  (`parcus eval --judged` family), off by default, only enabled on a held no-regression bar.
- Idempotent and fail-open; never elide the block a following turn's `tool_result` pairing needs.

## 6. Proposed slices (fidelity-first)

1. **Structured parse/serialize with byte-identical round-trip** for Anthropic + OpenAI tool
   shapes — **no optimization applied yet**. Pure fidelity + fail-open on unmodeled shapes. This is
   the foundation and the biggest correctness surface; land it with an exhaustive round-trip
   corpus before anything else.
2. **Apply existing tiers to structured content** — run the tiers over the `text` blocks within
   structured messages (immutable tool_use/tool_result/image blocks untouched). This is where
   tool-using traffic gets the wins already built. **Tier-0 lossless landed** (whitespace-normalise
   text blocks, code-fence-aware, verbatim otherwise); **Tier-1 filler landed** (strip allow-listed
   fillers from `text` blocks, immutable blocks untouched, model-free guardrail unchanged); **M1b
   cache injection landed** (mark the last content block of the breakpoint message with
   `cache_control`, and preserve any harness-supplied `cache_control` rather than adding a competing
   one). **Tier-2 learned landed** (reduce `text` blocks inside structured messages with the local
   reducer, immutable tool_use/tool_result/image blocks untouched; still gated offline by the
   answer-preservation judge, `ok=None` at runtime). All tiers now reach structured content.
3. **M1d tool-result elision** ✓ *(landed)* — a lossy, opt-in `ToolResultElider` compressor tier
   (`elide_tool_results`, off by default) that stubs stale `tool_result` payloads (older than
   `elide_keep_recent`), preserving `tool_use_id`/`is_error`. Size-gated so it only fires when the
   payload exceeds the stub (never-cost-more), fail-open; validate on `eval --judged` before use.

Each slice is independently shippable and reviewable; slice 1 carries the risk and must be
over-tested before slices 2–3 build on it.

## 7. Testing

- **Round-trip fidelity (property-based):** a Hypothesis corpus of realistic Anthropic/OpenAI
  structured bodies (mixed text/tool_use/tool_result/image, multi-turn) → assert
  `serialize(parse(b)) == b` for unmodified requests, and that any unmodeled shape yields `None`
  (passthrough). This is the primary safety net.
- **Fail-open fault injection** at the new parser seam (malformed blocks, missing ids, unknown
  types) → passthrough, never raise.
- **Optimization correctness** (slice 2): immutable blocks (tool_use/image) reproduced byte-for-byte;
  only text blocks change.
- **Elision answer-preservation** (slice 3): the `eval --judged` no-regression bar; a scenario in
  `qa/cache-inject/`-style harness measuring history-token reduction vs. task quality.

## 8. Security & privacy

- Block content (esp. `tool_result`) is **untrusted** and may carry secrets/PII → redaction still
  applies before any persist/log (master §5, `std-owasp-llm` LLM02); never render/execute block
  content. Elision stubs must not leak elided content into logs.
- The parser treats every field as untrusted input; malformed structures fail closed to passthrough.

## 9. Risks & open questions

- **Fidelity is everything** — the whole expansion is only as safe as the round-trip corpus. A
  narrow, well-tested subset that passes through the rest is far better than broad coverage that
  occasionally corrupts. Widen deliberately.
- **Model-shape churn:** Anthropic/OpenAI evolve block shapes; unmodeled fields must round-trip via
  a verbatim "carry-through" of unknown keys, or trigger passthrough — decide which per block type.
- **Where elision state lives:** stateless per-request (elide by position/recency in the turn) vs.
  memory-tier-aware — start stateless (position-based `keep_recent`), like the current compaction.
- **Interaction with M1b/M1c:** injection breakpoint placement and history compaction must be
  recomputed over the block model, not the string model.

## 10. Recommendation

Treat slice 1 (round-trip fidelity) as its own PR with an exhaustive property-test corpus and
**zero behavior change** (parse+serialize identity, still no optimization) — prove fidelity in
isolation. Only then layer slices 2–3. Do **not** bundle the parser expansion with the lossy
elision; the correctness surfaces are different and the elision needs the eval gate.
