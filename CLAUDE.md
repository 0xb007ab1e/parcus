# parsimony — project rules

> Inherits the master SSDLC ruleset (`~/.claude/CLAUDE.md`) automatically.
> A local-first, token-thrift **inference proxy** for agentic harnesses. See `PLAN.md`.

## Applied rule modules
@~/.claude/rules/lang-python.md
@~/.claude/rules/std-owasp-api.md            # the proxy IS an API edge
@~/.claude/rules/std-owasp-llm.md            # relays LLM traffic; treat model/tool/cached content as untrusted
@~/.claude/rules/std-owasp-proactive.md
@~/.claude/rules/std-cwe.md
@~/.claude/rules/topic-token-optimization.md # the core domain
@~/.claude/rules/topic-caching.md            # exact-hash response cache, TTL, invalidation
@~/.claude/rules/topic-api-consumption.md    # upstream providers are untrusted/unreliable
@~/.claude/rules/topic-architecture-patterns.md  # ports & adapters / functional core (hosted-ready)
@~/.claude/rules/topic-dependency-injection.md
@~/.claude/rules/topic-error-handling.md     # FAIL OPEN: forward original request on any uncertainty
@~/.claude/rules/topic-reliability.md        # timeouts, circuit-break upstreams, streaming
@~/.claude/rules/topic-realtime.md           # SSE streaming passthrough
@~/.claude/rules/topic-logging-observability.md  # redaction + savings metrics
@~/.claude/rules/topic-database.md           # SQLite + sqlite-vec store
@~/.claude/rules/std-privacy.md              # cached prompts/responses may hold PII
@~/.claude/rules/std-supplychain.md
@~/.claude/rules/workflow-cicd.md
@~/.claude/rules/workflow-threat-model.md
@~/.claude/rules/workflow-vuln-mgmt.md
@~/.claude/rules/workflow-cve-management.md
@~/.claude/rules/topic-testing.md
# @~/.claude/rules/topic-multi-tenancy.md    # DEFER: only when the hosted/shared mode is built
# @~/.claude/rules/topic-authn-authz.md      # DEFER: hosted mode (proxy auth)

## Stack
- Runtime: Python 3.12+; packaging via `pyproject.toml` (uv/pip), lockfile + hashes.
- Proxy: FastAPI + httpx (async, streaming). Tokenizers: `tiktoken` (OpenAI) + Anthropic count.
- Store: SQLite + `sqlite-vec`; `networkx` for graph algorithms. Optional local models lazy-loaded.
- Validation: pydantic v2 at every boundary.

## Project-specific rules (deviations & emphases, with reasons)
- **FAIL OPEN for availability (deliberate deviation from the usual fail-closed default).**
  The proxy's job is to be invisible. On ANY uncertainty — unknown route, parse failure,
  compressor error, store error — forward the **original, unmodified** request upstream and
  serve the real response. Reason: a token optimizer that breaks a harness or changes a result
  is worse than useless. Security decisions still fail *closed* (e.g. refuse to log/persist a
  secret); only the *optimization* path fails open. (`topic-error-handling`)
- **Correctness is the gate; tokens are the objective.** No lossy transform ships unless it
  holds a measured no-regression bar on the eval set (master §4). Critical path = the
  request-transform + cache-decision pipeline → 100% coverage (`topic-testing`).
- **Local-only models.** Saving tokens by issuing other inference calls is self-defeating.
  Heuristics are model-free; embedding/learned compressors are LOCAL, lazy, opt-in. Never an
  outbound call to optimize.
- **Provider API keys are the crown jewels.** From env/secret store only, never logged, never
  cached, never in VCS (`workflow-secrets`, master §1). Bind loopback + tailnet, never public
  (`topic-tailnet-dev-access`).
- **Cache/graph data = confidential.** Redact secrets/PII before persist; default TTL; opt-out
  patterns (paths/regex) and a kill switch; optional at-rest encryption (M2+) (master §5).
- **Never modify provider responses.** We compress *requests* and serve *exact* cache hits only;
  response bytes are passed through untouched.
- Data classification: **confidential** (prompts/responses may contain source code, secrets, PII).
