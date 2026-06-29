#!/usr/bin/env bash
# HTTP-edge conformance / negative-fuzz harness for parcus.
#
# Fires schemathesis-generated requests (conforming AND malformed/boundary, per openapi.yaml) at a
# running `parcus serve` pointed at a mock upstream, and asserts the single property that matters
# for a transparent proxy: it NEVER returns a 5xx (`not_a_server_error`). parcus must fail open
# (forward/handle) or return a deliberate 4xx/502 — never crash on bad input.
#
# Requires the project venv (`make setup`). schemathesis is run ephemerally and is NOT a project
# dependency — by default via `uvx schemathesis`; override for a local install:
#   SCHEMATHESIS="$PWD/.venv/bin/schemathesis" qa/fuzz/run.sh
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Split the (possibly multi-word) runner command on spaces explicitly — IFS above omits space.
IFS=' ' read -ra ST <<< "${SCHEMATHESIS:-uvx schemathesis}"
IFS=$'\n\t'

PARCUS_PY="${PARCUS_PY:-$REPO_ROOT/.venv/bin/python}"
MOCK_PORT="${MOCK_PORT:-8993}"
PROXY_PORT="${PROXY_PORT:-8789}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"

[ -x "$PARCUS_PY" ] || { echo "venv python not found at $PARCUS_PY (run make setup)" >&2; exit 2; }

mock_pid=""
proxy_pid=""
cleanup() {
  [ -n "$proxy_pid" ] && kill "$proxy_pid" 2>/dev/null || true
  [ -n "$mock_pid" ] && kill "$mock_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "→ starting fuzz mock upstream on :$MOCK_PORT"
"$PARCUS_PY" "$SCRIPT_DIR/mock_upstream.py" --port "$MOCK_PORT" >/dev/null 2>&1 &
mock_pid=$!

echo "→ starting parcus on :$PROXY_PORT (both upstreams → mock)"
PARCUS_ANTHROPIC_UPSTREAM="http://127.0.0.1:$MOCK_PORT" \
PARCUS_OPENAI_UPSTREAM="http://127.0.0.1:$MOCK_PORT" \
  "$PARCUS_PY" -m parcus.cli serve --host 127.0.0.1 --port "$PROXY_PORT" >/dev/null 2>&1 &
proxy_pid=$!

for _ in $(seq 1 50); do
  curl -fsS -o /dev/null "http://127.0.0.1:$PROXY_PORT/__parcus__/health" 2>/dev/null && break
  sleep 0.2
done

echo "→ fuzzing the edge ($MAX_EXAMPLES examples/op, positive+negative; check: not_a_server_error)"
"${ST[@]}" run "$SCRIPT_DIR/openapi.yaml" \
  --url "http://127.0.0.1:$PROXY_PORT" \
  --checks not_a_server_error \
  --mode all \
  --max-examples "$MAX_EXAMPLES"
echo "→ done (no server errors = parcus failed open on every input)"
