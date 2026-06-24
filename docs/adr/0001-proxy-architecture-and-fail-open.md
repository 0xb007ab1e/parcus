# ADR 0001 — Transparent proxy, ports & adapters, fail-open

- Status: Accepted
- Date: 2026-06-24
- Deciders: project author (planning Q&A)

## Context

We need tooling that reduces tokens spent per turn across multiple agentic harnesses
(Claude Code, pi, opencode) without per-harness integration, while never degrading task
correctness. Two levers: compress outbound requests, and avoid redundant outbound calls.

## Decision

1. **Transparent inference proxy/gateway** as the integration model — speaks the Anthropic
   Messages API and OpenAI Chat Completions, with transparent pass-through for anything else.
   Harnesses just repoint their base URL. One implementation benefits all harnesses.
2. **Ports & adapters / functional-core architecture.** Pure, provider-agnostic transform &
   cache-decision logic; I/O (FastAPI ingress, httpx upstream, SQLite store, tokenizers,
   optional local models) behind injected ports. Keeps the core testable without mocks and
   keeps a future hosted/multi-tenant mode a matter of swapping adapters.
3. **Fail open for availability.** On any uncertainty in the optimization path (unknown route,
   parse failure, compressor/store error) forward the **original, unmodified** request and
   serve the real response. Security checks still fail closed.
4. **Correctness is the gate.** No lossy transform or non-exact cache reuse ships without a
   measured no-regression result on an eval set. Exact/normalized-hash cache is the default.
5. **Local-only models; local-first deployment.** Never make an inference call to save one;
   bind loopback + tailnet, never public.

## Consequences

- (+) Universal harness support, no vendor lock to one tool; clean path to a hosted mode later.
- (+) The proxy can never break a harness or silently change a result — the worst failure is
  "no savings this turn."
- (+) Pure core → high-confidence tests, 100% coverage achievable on the critical pipeline.
- (−) Proxy holds provider API keys → a high-value credential boundary to defend (see threat
  model); mitigated by never logging/caching keys and least privilege.
- (−) Parsing each provider dialect into a canonical model is ongoing maintenance as APIs
  evolve; mitigated by pass-through fallback for anything unrecognized.

## Alternatives considered

- **Library/SDK or CLI preprocessor** — rejected as the primary model (needs per-harness wiring
  / is coarse-grained); the core logic is still importable as a library.
- **MCP server** — rejected as primary (agent must choose to call it, itself costing tokens);
  may be offered later as an optional surface.
- **Aggressive semantic cache** returning "close enough" answers — rejected as default
  (correctness risk in stateful agentic loops); available opt-in with strict thresholds.
