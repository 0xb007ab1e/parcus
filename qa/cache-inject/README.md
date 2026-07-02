# qa/cache-inject — validate M1b prompt-cache injection

Confirms that parcus's **prompt-cache injection** (M1b) actually makes an explicit-breakpoint
provider (Anthropic) serve a large re-sent prefix from its **prompt cache**, using the provider's
own `cache_read_input_tokens` as ground truth. This is the validation gate the roadmap flags for
M1b — see `docs/design/token-reduction-roadmap.md` §2.1 and `docs/validation/RESULTS.md`.

Out-of-band, like the rest of `qa/`: nothing here is a project dependency or runs in CI. It only
imports `parcus` (+ stdlib); the live run also uses parcus's own `httpx` upstream.

## What it does

Sends the *same* large-prefix request through parcus **twice** per condition and compares:

| condition | forwarded request | expected turn-2 `cache_read` |
|---|---|---|
| **baseline** (`cache_inject` off) | no `cache_control` | ~0 (provider caches nothing) |
| **inject on** (`cache_inject` on) | `cache_control` on the stable prefix | ≈ prefix tokens (served from cache) |

Compression is disabled (`NullCompressor`) and parcus's own response cache is off, so the *only*
variable between conditions is the injected breakpoint — the turn-2 `cache_read` delta is the win.

## Run it

**Offline harness check** (no key, no network) — proves parcus injects `cache_control` only when
enabled and that the harness reads usage correctly, against a fake Anthropic upstream that caches
only when it receives a `cache_control` marker:

```sh
.venv/bin/python qa/cache-inject/validate.py --self-test
```

**Live validation** (your Anthropic key; nothing is printed but token counts):

```sh
ANTHROPIC_API_KEY=... .venv/bin/python qa/cache-inject/validate.py \
  [--model claude-haiku-4-5] [--prefix-tokens 5000]
```

Four cheap calls total (2 conditions × 2 turns), tiny `max_tokens`. Default model is a low-cost
one (`claude-haiku-4-5`); override with `--model` or `PARCUS_VALIDATE_MODEL`. The stable prefix is
sized past the 4096-token Anthropic injection floor.

## Interpreting the result

A turn-2 `cache_read` that is ~0 in baseline and ≈ prefix-tokens with injection on confirms M1b
delivers the provider-cache win end-to-end. Record the numbers in `docs/validation/RESULTS.md`
and only then consider changing the `cache_inject` default (it ships **off** until validated — a
malformed `cache_control` would 400 a live request).
