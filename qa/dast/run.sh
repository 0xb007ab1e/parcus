#!/usr/bin/env bash
# DAST: OWASP ZAP baseline (passive) scan of a running parcus.
#
# Stands up a mock upstream + `parcus serve` and points ZAP's baseline scan at the proxy. The
# baseline is a *passive* scan: it inspects the responses parcus actually emits (health, errors,
# cache replays, relayed bodies) for information leakage and missing/incorrect security headers —
# it does not attack. Active API scanning of the request surface is covered by schemathesis
# (qa/fuzz); ZAP complements it on the response side.
#
# Requires Docker (or set DOCKER=podman) and the project venv. ZAP runs as a container — it is NOT
# a project dependency. The ~1.5 GB image is pulled on first run.
#
# Usage:  qa/dast/run.sh                 # scan the proxy, report to qa/dast/report.html
#         DOCKER=podman qa/dast/run.sh
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DOCKER="${DOCKER:-docker}"
ZAP_IMAGE="${ZAP_IMAGE:-ghcr.io/zaproxy/zaproxy:stable}"
PARCUS_PY="${PARCUS_PY:-$REPO_ROOT/.venv/bin/python}"
MOCK_PORT="${MOCK_PORT:-8994}"
PROXY_PORT="${PROXY_PORT:-8790}"

command -v "$DOCKER" >/dev/null 2>&1 || { echo "$DOCKER not found (set DOCKER=...)" >&2; exit 2; }
[ -x "$PARCUS_PY" ] || { echo "venv python not found at $PARCUS_PY (run make setup)" >&2; exit 2; }

mock_pid=""
proxy_pid=""
cleanup() {
  [ -n "$proxy_pid" ] && kill "$proxy_pid" 2>/dev/null || true
  [ -n "$mock_pid" ] && kill "$mock_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "→ starting mock upstream on :$MOCK_PORT"
"$PARCUS_PY" "$REPO_ROOT/qa/load/mock_upstream.py" --port "$MOCK_PORT" >/dev/null 2>&1 &
mock_pid=$!

echo "→ starting parcus on :$PROXY_PORT (upstreams → mock)"
PARCUS_ANTHROPIC_UPSTREAM="http://127.0.0.1:$MOCK_PORT" \
PARCUS_OPENAI_UPSTREAM="http://127.0.0.1:$MOCK_PORT" \
  "$PARCUS_PY" -m parcus.cli serve --host 127.0.0.1 --port "$PROXY_PORT" >/dev/null 2>&1 &
proxy_pid=$!

for _ in $(seq 1 50); do
  curl -fsS -o /dev/null "http://127.0.0.1:$PROXY_PORT/__parcus__/health" 2>/dev/null && break
  sleep 0.2
done

echo "→ running ZAP baseline (passive) against the proxy"
# --network host so the container reaches the loopback-bound proxy. -I = don't fail the run on
# warnings (we review the report). -c applies the rule tuning in zap-baseline.conf.
"$DOCKER" run --rm --network host \
  -v "$SCRIPT_DIR:/zap/wrk:rw" \
  "$ZAP_IMAGE" zap-baseline.py \
  -t "http://127.0.0.1:$PROXY_PORT/__parcus__/health" \
  -c zap-baseline.conf \
  -r report.html \
  -I || echo "  (ZAP reported findings — review qa/dast/report.html)"
echo "→ done — report at qa/dast/report.html"
