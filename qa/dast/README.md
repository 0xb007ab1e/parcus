# DAST — OWASP ZAP baseline scan

A **passive** dynamic scan of a running `parcus serve`. ZAP inspects the responses parcus emits
(health/stats endpoints, error responses, relayed bodies) for information leakage and missing or
incorrect security headers — it does not attack. It complements the **active request-edge fuzzing
in `qa/fuzz/`** (schemathesis): that one hammers what parcus *accepts*; this one reviews what
parcus *returns*. Together with the in-process `tests/integration/test_credential_safety.py`
(credential never leaks into metrics/headers/cache), they cover the proxy's security surface.

## Run

```bash
qa/dast/run.sh                 # needs Docker; pulls ghcr.io/zaproxy/zaproxy on first run
DOCKER=podman qa/dast/run.sh   # or podman
```

The runner stands up the mock upstream + the proxy (loopback), runs the ZAP baseline container
against the proxy with `--network host`, writes `report.html`, and tears down. `zap-baseline.conf`
tunes the ruleset: HTML-only browser-header rules (CSP, anti-clickjacking, HSTS, Permissions-
Policy) are **N/A to a header-less API proxy** and downgraded with justification, while
information-leak / error-disclosure rules stay at ZAP's default so genuine issues still surface.

## Posture

ZAP runs as a container — **not** a project dependency; nothing added to `pyproject`. Add-only
under `qa/dast/`. Not wired into CI by default (it pulls a large image and stands up a live
server + the proxy); run on demand or as a scheduled security job. Review `report.html`; treat
any non-ignored WARN/FAIL as a finding to trace (master §7 severity).

> Note: ZAP is lower-signal on a transparent proxy than on a web app — most of its value here is
> confirming parcus's *own* responses (errors, endpoints) don't leak internals. For request-side
> robustness, `qa/fuzz/` (schemathesis) is the primary tool.
