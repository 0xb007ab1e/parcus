# Lossy-tier quality gate — active-session judge (no API key)

**Option A** of the validation plan: grade parcus's lossy compression with a *real* model — the
**active Claude session itself** — instead of a deterministic substring judge or a paid API key.

The synthetic `KeywordRecallJudge` (`parcus eval --judged`) only checks substring recall; this
harness gets a real model to rule on **answer preservation**: would the compressed prompt elicit
the same answer — was any instruction, constraint, number, negation, ordering, or code/identifier
lost? It grades parcus's **actual** Tier-1/Tier-2 output on a realistic agentic corpus.

## Flow

1. **`python qa/judge/build_cases.py`** — compresses the corpus through the real lossy tiers
   (L1 default filler, L2 aggressive filler, L3 learned when a local model is present) and writes
   `cases.json` (each `original → compressed` pair + token counts). Run with the local learned
   model for L3:
   ```bash
   PARCUS_LEARNED_MODEL=$HOME/models/gpt2 TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python qa/judge/build_cases.py
   ```
2. **The active Claude session judges** each case and writes `verdicts.json`
   (`{id: {preserved: bool, reason: str}}`). This is the "use the active session" step — the
   model is the session, in-conversation; no key, no external call.
3. **`python qa/judge/score.py`** — joins cases + verdicts into `RESULTS.md`: per-tier mean
   reduction + preservation rate, with any answer-changing case listed as a regression
   (non-zero exit).

`cases.json`, `verdicts.json`, and `RESULTS.md` are committed as the recorded validation
artifact (re-run to refresh).

## Scope & honesty

- **Legitimate use of the session.** The session is used only for *offline evaluation*, never in
  the live compression path — parcus's "local-only models, never an outbound call to optimize"
  tenet is about the request path; eval is allowed (and expected) to use a real model.
- **Not the full validation.** This judges *answer preservation of the compressed request text*.
  It does **not** measure real token usage or provider prompt-cache interaction through parcus —
  that needs a real `ANTHROPIC_API_KEY` and provider-`usage` capture (the separate "Option B").
- **Learned-tier caveat:** on a short-prompt corpus the learned tier mirrors aggressive filler
  (LLMLingua keeps short spans whole), so L3 is only meaningfully exercised on long prose.

Latest run: **24/24 cases preserved** (see `RESULTS.md`).
