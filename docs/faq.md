# parcus — FAQ

Comprehensive answers, grouped by topic. See also the plain-language [guide](guide.md) and the
[technical reference](technical-reference.md).

## Basics

**What is parcus?**
A small proxy you run on your own machine that sits between an AI coding harness (Claude Code, pi,
opencode, …) and the model provider (Anthropic/OpenAI). It reduces the tokens spent per turn —
by trimming low-information text and by not re-paying for identical/near-identical calls — without
changing your results.

**Who is it for?**
Anyone running token-hungry agentic tools who wants to cut spend and latency. It's local-first
and single-user by default; there's an optional hosted/multi-tenant mode for teams.

**Why is it called "parcus"?**
`parcus` is Latin for *sparing/thrifty* — the literal root of *parsimony* (via *parcere*, "to
spare"). It captures both the goal (frugality with tokens) and the method (parsimony / Occam's
razor — keep only the minimal sufficient form). The project was originally "parsimony"; that name
is taken on PyPI, so it adopted its own root word. See the [README](../README.md#why-parcus).

**How much will it save?**
It depends entirely on your traffic. Lossless + exact cache are free, zero-risk wins but modest on
their own; the bigger savings come from the cache hit rate on repetitive agent loops and from
opting into filler/learned compression. Run `parcus eval` and `parcus stats` to measure *your*
numbers rather than trusting a headline figure — the project deliberately ships measurement, not
marketing claims.

## Correctness & trust

**Will it change the model's answers?**
The default tiers (lossless whitespace + exact cache) cannot change meaning. The opt-in tiers can
*in principle*, which is exactly why they're opt-in and gated: filler removal is proven (model-
free) to drop only allow-listed words; the learned compressor and near-duplicate cache must pass
an offline quality/precision gate before you enable them. The guiding rule is **correctness is the
gate; tokens are the objective.**

**What if parcus breaks or doesn't understand my request?**
It **fails open**: on any error or ambiguity it forwards your **original, unmodified** request and
returns the real response. The worst case is "no savings this turn," never a broken result. The
request-transform + cache-decision core is held to **100% test coverage**.

**How do I verify it's not degrading quality?**
`parcus eval` runs sample prompts through the compressors and reports token reduction plus a
correctness gate: model-free equivalence for the safe tiers, and a quality judge
(`--retrieval`/`--similarity`) for the lossy ones. You can point it at your own dataset.

**How does it know what *not* to touch?**
It classifies each prompt into spans; code, file paths, URLs, quoted text, tool/function JSON, and
your trailing instruction are **immutable** and reproduced verbatim. Only prose "filler" spans are
ever rewritten.

## Privacy & security

**Does it phone home or make its own AI calls?**
No. It never makes outbound calls except to forward your request to your chosen provider. Optional
compressors/embedders run **locally** — saving tokens by making other inference calls would defeat
the purpose.

**What happens to my provider API key?**
It's forwarded to the provider and otherwise **never logged, cached, or persisted**.

**Where is my data stored, and is it safe?**
The response cache and (optional) memory graph live in a `0600` SQLite file under `.parcus/`.
Secrets/PII are redacted before anything is persisted, and matching requests can be excluded with
no-cache patterns. For shared/backed-up hosts you can enable **AES-256-GCM at-rest encryption**.

**Can it be exposed to the internet?**
No — it refuses to bind `0.0.0.0`/public. It's meant for loopback or a private tailnet
(`http://<host>:<port>`), reachable from your own devices, gated by your tailnet ACLs.

**Tell me about the at-rest encryption.**
Response bodies are sealed with AES-256-GCM (an authenticated cipher), the cache key bound as
additional authenticated data (tamper/relocation fail the auth tag). The key comes from env or a
keyfile (never VCS); enabling encryption without a valid key **fails closed** at startup. It
supports **graceful key rotation** (current + previous keys) and, in hosted mode, **per-tenant
keys** with **crypto-shredding** (withhold a tenant's key to erase their cached data instantly).

## Features & configuration

**What providers/formats are supported?**
Anthropic Messages and OpenAI Chat Completions (a text-only subset is canonicalized for
compression/caching; richer payloads pass through untouched). Anything else is proxied verbatim.

**Does streaming work?**
Yes — streaming requests are transparently passed through (no compression/cache applied to them in
this version), routed to the correct provider and streamed back unmodified.

**How do I turn features on?**
Set `PARCUS_*` environment variables (or a `.env`). Everything beyond Tier-0 + exact cache is off
by default. See the [config reference](technical-reference.md#7-configuration) and
[`.env.example`](../.env.example). A safe progression: `FILLER` → (validate) `FILLER_AGGRESSIVE` /
`LEARNED` → `SIMILARITY_CACHE` → `MEMORY` + `MEMORY_INJECT`.

**What's the difference between the cache types?**
The **exact** cache replays a response only for a byte-identical request. The **similarity** cache
(opt-in) can replay for a *near-duplicate* request (same model + tenant, cosine ≥ a high
threshold), trading a little correctness risk for more hits — so it's gated by a no-false-hit
precision check.

**Why does enabling the similarity cache default to a "local" embedder?**
Because the dependency-free *lexical* embedder can't tell apart requests that differ only in
numbers/entities ("10 replicas" vs "2") — unsafe for serving cached answers. The safe default is a
local semantic model; using the lexical one requires an explicit `SIMILARITY_ALLOW_LEXICAL=true`.

**What is the memory graph for?**
To avoid re-sending large context every turn: it keeps durable facts/decisions and either injects
only the *relevant* subset (Track B) or replaces old turns with a rolling summary (Track C). It's
off by default and behind a retrieval-recall gate because it changes what the model sees.

## Operations

**How do I see what it's doing?**
`parcus stats` (and `GET /__parcus__/stats` for JSON, `/__parcus__/metrics` for Prometheus,
`/__parcus__/health` for liveness) report per-stage token reduction and accuracy (the model-free
invariant pass-rate) plus offline eval-gate scores. Metrics are content-free counts.

**What's "hosted/multi-tenant mode"?**
An opt-in mode to run one shared instance for several principals. The tenant is derived
server-side from the inbound credential (never a client field), and per-tenant **cache and memory
are isolated**, with an optional **edge allow-list** (401 for unlisted) and **per-tenant rate
limits** (429 + Retry-After). Single users can ignore it.

**Does it add latency?**
The local transforms are fast and bounded; a cache hit is far cheaper than a provider round-trip.
Heavy local models (Tier-2 learned, local embeddings) are opt-in. Compression cost is itself a
guarded trade-off — if it ever exceeds the savings, that's a tuning signal.

**Is it production-ready / on PyPI?**
v0.1.0 is the first tagged release, published as a signed GitHub Release with an SBOM and SLSA
provenance. PyPI publishing is wired (Trusted Publishing) but gated until the publisher is
configured. It's local-first tooling; the hosted mode adds the controls needed for shared use.

## The name, once more

**Is there a deeper meaning?**
`parcus` → *parsimony* → Occam's razor: *entia non sunt multiplicanda praeter necessitatem* —
"entities must not be multiplied beyond necessity." Swap "entities" for "tokens" and you have the
product spec. The tool is the razor; your prompt keeps only what carries meaning.

---

*Didn't find your question? See the [technical reference](technical-reference.md), the
[ADRs](adr/) for design rationale, or open an issue.*
