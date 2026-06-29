# Load / latency-overhead / soak testing (k6)

Measures **parcus's own per-request overhead** and its behavior under sustained load. parcus must
be invisible — its processing (parse → compress → cache decision → re-serialize → forward) should
add only a small, bounded latency, and it must not leak memory over a long session.

To isolate parcus, the harness runs it against a **mock upstream** (`mock_upstream.py`) instead of
a real provider — no network, no key, no provider latency. It then compares three phases:

| phase | path | what it shows |
|---|---|---|
| `direct` | client → mock | baseline |
| `proxy-miss` | client → parcus → mock | full pipeline overhead |
| `proxy-hit` | client → parcus (cache replay) | the cache win (no upstream hop) |

**Overhead = proxy-miss − direct.** Proxy RSS is sampled before/after as a coarse leak signal.

## Prerequisites

- The project venv: `make setup`.
- **k6** (a single Go binary, **not** a project dependency): install from <https://k6.io/docs/get-started/installation/>,
  or drop the binary anywhere and point the runner at it with `K6=/path/to/k6`.

## Run

```bash
qa/load/run.sh                      # quick: 10 VUs × 10s per phase
VUS=50 DURATION=2m qa/load/run.sh   # heavier
DURATION=10m qa/load/run.sh         # soak — watch the before/after RSS for leaks
K6=./k6 qa/load/run.sh              # use a local k6 binary not on PATH
```

Tunables (env): `VUS`, `DURATION`, `MOCK_PORT`, `PROXY_PORT`, `PARCUS_PY`, `K6`.

The runner starts the mock and `parcus serve` (bound to loopback), waits for readiness, runs the
three k6 phases, prints a markdown table of avg/med/p95/p99 + throughput, the overhead delta, and
the RSS change — then tears everything down.

## Interpreting

- **Overhead** should be small and stable; a large or growing p95 gap is a regression to profile
  (`topic-performance`). Treat the p95 overhead as a budgeted SLI.
- **proxy-hit** should be markedly faster than proxy-miss (it skips the upstream hop) — that is
  the cache earning its keep.
- **RSS** roughly flat over a soak = no obvious leak; steady growth ⇒ investigate
  (`topic-resource-management`; ties to the unclosed-resource hygiene in the backlog).

Streaming (SSE) passthrough is exercised separately (see the SSE-fidelity harness); this harness
covers the buffered, cacheable path where parcus does the most per-request work.

## Notes

k6 is invoked as an external binary — nothing is added to `pyproject.toml`. Not wired into CI by
default (load tests are environment-sensitive); run on demand or as a scheduled/nightly job, and
pin the k6 version there.
