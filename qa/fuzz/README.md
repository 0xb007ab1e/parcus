# HTTP-edge conformance / negative fuzzing (schemathesis)

parcus's HTTP edge is an untrusted-input boundary (`std-cwe`, the threat model): clients send
arbitrary JSON, headers, and content-types. The proxy must **fail open on every input** — parse
and forward what it understands, pass through what it doesn't, return a deliberate 4xx/502 when it
must — but **never** crash with a 5xx. This harness proves that by fuzzing the running proxy.

`schemathesis` generates requests from `openapi.yaml` (both **conforming** and, in `--mode all`,
**malformed/boundary/negative**) and fires them at a live `parcus serve`. The one check we run is
**`not_a_server_error`**: any 5xx is a failure. (Response-schema conformance is intentionally not
checked — parcus is a transparent proxy, not the API; `openapi.yaml` is only a generator seed.)

To keep fuzz traffic off real providers, `parcus serve` is pointed at `mock_upstream.py`.

## Run

```bash
# schemathesis is run ephemerally — NOT a project dependency.
qa/fuzz/run.sh                                            # via `uvx schemathesis` (needs uv)
SCHEMATHESIS="$PWD/.venv/bin/schemathesis" qa/fuzz/run.sh # or a local install
MAX_EXAMPLES=500 qa/fuzz/run.sh                           # deeper fuzz
```

Tunables (env): `MAX_EXAMPLES`, `MOCK_PORT`, `PROXY_PORT`, `PARCUS_PY`, `SCHEMATHESIS`.

The runner stands up the mock and the proxy (bound to loopback, both upstreams → mock), waits for
readiness, runs schemathesis against the proxy URL, and tears everything down. A non-zero exit
means schemathesis found a server error — i.e. a fail-open gap to fix.

## Notes

`schemathesis` is invoked as an external/ephemeral tool — nothing added to `pyproject.toml`. Not
wired into CI by default (it stands up a live server); run on demand or as a scheduled job, and
pin the schemathesis version there. Extend `openapi.yaml` as the accepted edge surface grows.

## Coverage-guided parser fuzzing (Atheris)

Where schemathesis fuzzes the proxy **black-box over HTTP**, `fuzz_parsers.py` fuzzes the parser
**functions directly** with [Atheris](https://github.com/google/atheris) (libFuzzer) — coverage-
guided, so it reaches edge cases black-box traffic rarely hits. It feeds mutated input to
`dialects.detect` / `dialects.parse`, `app._is_stream`, `tenant.derive_tenant`, and
`Redactor.has_secret`, letting any exception propagate so libFuzzer records it as a crash. These
must **fail open** (the engine calls `parse()` un-guarded, so a parser exception would become a
5xx).

```bash
.venv/bin/python -m pip install atheris    # not a project dependency; needs clang
qa/fuzz/run_atheris.sh                      # 30s campaign
MAX_TOTAL_TIME=300 qa/fuzz/run_atheris.sh   # longer
```

A `crash-*` file appearing here (gitignored) is a fail-open gap: the byte string in it made a
parser raise — reproduce with `python qa/fuzz/fuzz_parsers.py qa/fuzz/crash-<hash>`. Last run:
~978k inputs in ~26 s, 0 crashes.
