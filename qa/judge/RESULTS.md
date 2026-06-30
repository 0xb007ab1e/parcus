# parcus lossy-tier quality — active-session judge

Each case is parcus's **real** lossy-tier output for a realistic agentic prompt, judged by the active Claude session for **answer preservation** (would the compressed prompt elicit the same answer — was any instruction / constraint / number / negation / ordering / code lost?). Token counts are the heuristic tokenizer's. No API key used; the model is the session itself.

| tier | cases | preserved | mean reduction |
|---|--:|--:|--:|
| L1-filler | 8 | 8/8 | 13.8% |
| L2-aggressive | 8 | 8/8 | 17.4% |
| L3-learned | 8 | 8/8 | 17.4% |

**Overall: 24/24 cases preserved** (PASS — no answer-changing compression observed).

No regressions: across all judged cases the lossy tiers removed only discourse fillers / intensifiers; every load-bearing token (numbers, negations, ordering, code spans, named identifiers) survived. This is real-model confirmation of the structural guardrail, on realistic prompts including the tricky cases (negation, ordered steps, hard numeric constraints).

> Caveat: on this short-prompt corpus the **learned** tier (L3) produced output identical to aggressive filler (L2) in every case — gpt2/LLMLingua keeps short spans whole — so this run thoroughly validates the filler tiers but does **not** exercise genuinely lossy learned compression (which only engages on long prose). A long-context corpus is needed to judge L3's lossy behaviour.
