# The parcus guide

*A plain-language book about what parcus is, why it exists, and how to use it. No prior
knowledge assumed — if you've used an AI coding assistant, you can read this.*

---

## Chapter 1 — The problem: you pay for the same words over and over

When you use an AI assistant (Claude Code, Cursor, a chatbot, an "agent"), every message you send
is turned into **tokens** — roughly, pieces of words — and you're billed per token, both for what
you send (*input*) and what you get back (*output*).

Here's the catch with *agentic* tools (the ones that take many steps to do a task): on **every
single turn**, they re-send a huge amount of text — the long system instructions, the list of
tools the model can use, the entire back-and-forth so far, and any files or context you pasted.
The model has no memory between calls, so the harness keeps re-sending everything.

Two kinds of waste pile up:

1. **Redundancy** — you already sent that exact context last turn, and the turn before. You pay
   for it again each time.
2. **Filler** — a lot of natural-language text carries no instruction value. "Could you please
   just go ahead and carefully…" says the same thing as "…". The model behaves identically with
   or without the padding, but you pay for every word.

Over a long session, that adds up to real money and slower responses.

## Chapter 2 — The idea: parsimony

**parcus** is named after *parsimony* — being sparing, using no more than necessary (it's the
Latin root, *parcus* = "thrifty"). The same idea shows up in science as **Occam's razor**: prefer
the simplest version that still works; cut away everything that isn't doing a job.

That's the whole philosophy: **send the fewest tokens that still mean the same thing.**

parcus is a small program that runs on your own computer and sits *between* your AI tool and the
AI provider (Anthropic, OpenAI). Your tool thinks it's talking to the provider; the provider
thinks it's talking to your tool. In the middle, parcus trims the waste — and if it's ever unsure
about anything, it gets out of the way and passes your request through untouched.

## Chapter 3 — How parcus saves tokens (in plain terms)

Two strategies:

**A. Make each message smaller.** parcus rewrites the *prose* parts of your prompt to say the
same thing in fewer words. It is careful in layers ("tiers"), from totally safe to more
aggressive:

- *Tidy up* (always on): collapse pointless whitespace and blank lines. This can't change
  meaning — it's like removing double spaces.
- *Drop filler* (you turn it on): remove a curated list of empty words — "please", "just",
  "basically", "obviously". parcus mathematically checks that it removed *only* those exact words
  and nothing else.
- *Smart trim* (you turn it on, runs a small AI model on your machine): cut low-value words a
  model judges unnecessary. Because this one is a judgment call, you validate it against a test
  set before trusting it.

Crucially, parcus **never touches the parts that must stay exact**: code, file paths, links,
quoted text, the tool definitions, and your actual final instruction. Those are reproduced
character-for-character.

**B. Don't pay for the same answer twice.** If parcus has seen an identical request before, it
just replays the saved answer instantly — no call to the provider, no tokens spent. (Optionally,
it can also reuse the answer for a *near-identical* request, or keep a compact "memory" of your
session so it doesn't re-send everything every turn.)

## Chapter 4 — The golden rule: it never breaks your tool

This is the most important promise. A tool that saves you tokens but occasionally corrupts a
request or changes an answer would be a disaster — you'd never trust it. So parcus is built to
**"fail open"**: whenever anything is uncertain — it doesn't recognize the request, can't parse
it, a step errors out — it abandons the optimization and forwards your **original request,
exactly as you sent it**. The worst case is "no savings this turn," never "broken result."

The flip side: anything *security*-related fails the other way ("fail closed") — e.g., if it spots
something that looks like a password, it refuses to store it.

## Chapter 5 — A typical session, narrated

1. You start parcus on your laptop: `parcus serve`.
2. You tell your AI tool to use `http://127.0.0.1:8787` instead of the provider's address.
3. You work as normal. Each turn:
   - Your tool sends its big request to parcus.
   - parcus tidies the prose, (optionally) drops filler, checks whether it's seen this exact
     request before, then forwards what's left to the provider.
   - The provider's answer comes straight back to your tool, unmodified.
   - parcus quietly notes how many tokens it saved and whether its safety checks held.
4. Later you run `parcus stats` and see, e.g., "lossless reduced input 8%, filler 4%, cache hit
   rate 12%, accuracy 100%."

You never had to think about it. That's the point — parcus is meant to be invisible.

## Chapter 6 — Knowing it's safe (how you can trust it)

You don't have to take the savings on faith. parcus ships an **eval harness**: it runs sample
prompts through the compressors and *measures* that meaning is preserved.

- For the safe tiers (tidy-up, filler), the check is **model-free** — it's a mathematical proof
  that only whitespace / only allow-listed words changed. No AI judgment needed.
- For the smart-trim tier and the near-duplicate cache, it uses a **quality gate**: it checks
  that the answer you'd get is still right (a "no false hits" / "answer preserved" test) before
  you enable them.

Run `parcus eval` (and `--filler`, `--similarity`, etc.) any time to see the numbers for
yourself.

## Chapter 7 — Which features to turn on

parcus is conservative by default. Here's a sensible progression:

| You want… | Turn on | Trade-off |
|---|---|---|
| Free, zero-risk savings | nothing — Tier-0 + exact cache are already on | none |
| A bit more, still safe | `filler` | drops curated filler words only |
| Maximum prose savings | `filler_aggressive`, then `learned` | validate with `parcus eval` first |
| Skip near-identical calls | `similarity_cache` | tiny chance of reusing a too-similar answer → keep the threshold high; validate |
| Don't re-send long histories | `memory` + `memory_inject` | changes what the model sees → behind a recall gate |

Everything is a `PARCUS_*` setting (or `.env` entry). Start safe, measure, then opt into more.

## Chapter 8 — Privacy and your keys

- Your **provider API key** is never logged, never saved, never put in the cache. parcus just
  forwards it.
- parcus runs **only on your machine** (or your private tailnet) — it refuses to expose itself to
  the public internet, and it never makes its *own* AI calls to "help" (that would cost tokens —
  the opposite of the point).
- The cache stores responses on disk; before storing anything, parcus scrubs obvious secrets, and
  you can encrypt the cache at rest (AES-256) if you're on a shared machine.

## Chapter 9 — Sharing one parcus with a team (hosted mode)

By default parcus is a personal, single-user tool. There's an optional **hosted mode** for
running one shared instance for several people:

- Each user is kept strictly separate — one person can never see another's cached data.
- You can restrict who's allowed to use the instance, and cap how fast each user can drive it.
- You can encrypt each user's cached data with its own key, and instantly "shred" (erase) a
  user's data by withholding that key.

If you're just one person on your own laptop, you can ignore this entirely.

## Glossary

- **Token** — a chunk of text (≈ ¾ of a word) that models read/write and that you're billed for.
- **Harness / agent** — the tool driving the model in a loop (e.g., Claude Code).
- **Proxy** — a middleman program that requests pass through.
- **Lossless / lossy** — lossless changes can't alter meaning (like removing double spaces); lossy
  ones might, so parcus gates them behind tests.
- **Cache** — a store of past answers, replayed instead of re-asking.
- **Fail open** — on any doubt, do nothing and pass the original through.
- **Parsimony / Occam's razor** — use the minimal sufficient form; cut the unnecessary.

---

*Next: the [Technical Reference](technical-reference.md) for the exact shapes and configuration,
or the [FAQ](faq.md) for specific questions.*
