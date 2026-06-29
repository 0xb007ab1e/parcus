# Lossy-tier differential eval (promptfoo)

The **correctness gate for parcus's lossy compression tiers**. The model-free invariants prove
Tier-0 (lossless) and the Tier-1 *structure* (only allow-listed fillers removed), but they cannot
prove that removing those words leaves the model's **answer** unchanged — that is a semantic
question only a real model can settle. This eval answers it: run prompts through a real model and
have an LLM judge confirm the compressed prompt still earns the original answer.

It is **not** part of CI (it needs a model and a key, and it costs tokens). Run it periodically,
and **before enabling a lossy tier** (`FILLER_AGGRESSIVE`, later `LEARNED`) in production.

## What it does

1. `build_dataset.py` runs **locally on parcus** (no model, no network) and writes `dataset.yaml`:
   for each base task, the **original** prompt and parcus's **compressed** prompt (the real
   Tier-0 → Tier-1-aggressive chain).
2. `promptfooconfig.yaml` runs both variants through a model and applies an `llm-rubric` judge
   that passes only if the answer satisfies the **original** intent. The `compressed` variant is
   the gate; the `original` variant is a sanity control (should always pass).

## Run it

```bash
# 1. (Re)generate the dataset from parcus — local, no key, deterministic.
python qa/eval/build_dataset.py

# 2. Provide the model + grader key (never commit it — workflow-secrets).
export ANTHROPIC_API_KEY=...        # or swap to OPENAI_API_KEY and edit the providers below

# 3. Run the graded eval, then browse results.
npx -y promptfoo@latest eval -c qa/eval/promptfooconfig.yaml
npx -y promptfoo@latest view
```

`promptfoo` is invoked **ephemerally via `npx`** — it is *not* a project dependency (nothing is
added to `pyproject.toml` or `package.json`). In CI/nightly, pin a version (`promptfoo@<x.y.z>`)
rather than `@latest`.

## Configure for your stack

- **Model under test** — edit `providers` in `promptfooconfig.yaml` (or `--providers ...` on the
  CLI) to the model your harness actually uses. Default is a small Claude model to bound cost.
- **Grader** — `defaultTest.options.provider` is the LLM judge; same provider by default so one
  key suffices. Point it at a strong model for stricter judging.
- **Tasks** — extend `_TASKS` in `build_dataset.py`; keep them synthetic (no secrets/PII —
  master §5). To gate the **learned** tier too, add it to the chain in `build_dataset.py`.

## Interpreting results

- **compressed FAIL** → that compression (the aggressive filler set, or the learned tier) changed
  the answer: tighten the allow-list / lower the learned ratio, or do not ship that tier.
- **original FAIL** → the rubric or grader model is mis-calibrated (the control should pass); fix
  the rubric before trusting the compressed verdicts.

Distinct from the **CI-able, model-free** quality gate (`parcus eval`, `KeywordRecallJudge`),
which runs every build without a model; this is the deeper, real-model check.
