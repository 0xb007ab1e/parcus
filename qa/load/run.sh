#!/usr/bin/env bash
# Load / latency-overhead / soak harness for parcus.
#
# Stands up a mock upstream and `parcus serve` pointed at it, then runs k6 in three phases and
# reports parcus's per-request overhead:
#   1. direct      client → mock                 (baseline)
#   2. proxy-miss  client → parcus → mock         (full pipeline: parse/compress/cache/forward)
#   3. proxy-hit   client → parcus (cache replay) (no upstream hop)
# Overhead = proxy-miss − direct. The hit phase shows the cache win. Proxy RSS is sampled before
# and after as a coarse leak signal (run with DURATION=10m for a real soak).
#
# Requires: k6 on PATH (or K6=/path/to/k6) and the project venv (`make setup`). No network, no
# provider key — the mock replaces the provider.
#
# Usage:  qa/load/run.sh                 # quick: 10 VUs × 10s per phase
#         VUS=50 DURATION=2m qa/load/run.sh
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

K6="${K6:-k6}"
PARCUS_PY="${PARCUS_PY:-$REPO_ROOT/.venv/bin/python}"
MOCK_PORT="${MOCK_PORT:-8991}"
PROXY_PORT="${PROXY_PORT:-8788}"
VUS="${VUS:-10}"
DURATION="${DURATION:-10s}"
OUT_DIR="$(mktemp -d)"

command -v "$K6" >/dev/null 2>&1 || { echo "k6 not found (set K6=/path/to/k6); see README" >&2; exit 2; }
[ -x "$PARCUS_PY" ] || { echo "venv python not found at $PARCUS_PY (run make setup)" >&2; exit 2; }

mock_pid=""
proxy_pid=""
cleanup() {
  [ -n "$proxy_pid" ] && kill "$proxy_pid" 2>/dev/null || true
  [ -n "$mock_pid" ] && kill "$mock_pid" 2>/dev/null || true
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

wait_ready() {  # $1=url $2=label
  for _ in $(seq 1 50); do
    if curl -fsS -o /dev/null "$1" 2>/dev/null; then return 0; fi
    sleep 0.2
  done
  echo "timed out waiting for $2 ($1)" >&2
  exit 1
}

echo "→ starting mock upstream on :$MOCK_PORT"
"$PARCUS_PY" "$SCRIPT_DIR/mock_upstream.py" --port "$MOCK_PORT" >"$OUT_DIR/mock.log" 2>&1 &
mock_pid=$!

echo "→ starting parcus on :$PROXY_PORT (upstream → mock)"
PARCUS_ANTHROPIC_UPSTREAM="http://127.0.0.1:$MOCK_PORT" \
  "$PARCUS_PY" -m parcus.cli serve --host 127.0.0.1 --port "$PROXY_PORT" \
  >"$OUT_DIR/proxy.log" 2>&1 &
proxy_pid=$!

# The mock has no GET route; probe its TCP port via the proxy's health endpoint + a direct POST.
wait_ready "http://127.0.0.1:$PROXY_PORT/__parcus__/health" "parcus"
curl -fsS -o /dev/null -X POST "http://127.0.0.1:$MOCK_PORT/v1/messages" \
  -H 'content-type: application/json' -d '{}' || { echo "mock not responding" >&2; exit 1; }

rss() { ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }
rss_before="$(rss "$proxy_pid")"

run_phase() {  # $1=label $2=target $3=mode
  echo "→ k6 phase: $1 (VUS=$VUS DURATION=$DURATION)"
  TARGET="$2" MODE="$3" VUS="$VUS" DURATION="$DURATION" \
    "$K6" run --quiet --summary-export "$OUT_DIR/$1.json" "$SCRIPT_DIR/load_test.js" \
    >"$OUT_DIR/$1.out" 2>&1 || echo "  (phase $1 had threshold failures — see numbers below)"
}

run_phase direct     "http://127.0.0.1:$MOCK_PORT"  miss
run_phase proxy-miss "http://127.0.0.1:$PROXY_PORT" miss
run_phase proxy-hit  "http://127.0.0.1:$PROXY_PORT" hit

rss_after="$(rss "$proxy_pid")"

PARCUS_RSS_BEFORE="$rss_before" PARCUS_RSS_AFTER="$rss_after" "$PARCUS_PY" - "$OUT_DIR" <<'PY'
import json, sys, os
out = sys.argv[1]
def load(name):
    with open(os.path.join(out, name + ".json")) as f:
        return json.load(f)["metrics"]
def g(m, *keys):
    for k in keys:
        if k in m: return m[k]
    return None
rows = []
for label in ("direct", "proxy-miss", "proxy-hit"):
    m = load(label)
    d = m.get("http_req_duration", {})
    reqs = m.get("http_reqs", {})
    rows.append((label, g(d, "avg"), g(d, "med"), g(d, "p(95)"), g(d, "p(99)", "p(95)"), g(reqs, "rate"), g(reqs, "count")))
print("\n## parcus load test\n")
print(f"| phase | avg ms | med ms | p95 ms | p99 ms | req/s | reqs |")
print(f"|---|--:|--:|--:|--:|--:|--:|")
for label, avg, med, p95, p99, rate, cnt in rows:
    f = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"
    print(f"| {label} | {f(avg)} | {f(med)} | {f(p95)} | {f(p99)} | {f(rate)} | {int(cnt) if cnt else 0} |")
direct = rows[0][3]; miss = rows[1][3]
if isinstance(direct, (int,float)) and isinstance(miss, (int,float)):
    print(f"\n**Proxy overhead (p95): {miss - direct:+.2f} ms** (proxy-miss − direct).")
rb, ra = os.environ.get("PARCUS_RSS_BEFORE"), os.environ.get("PARCUS_RSS_AFTER")
if rb and ra:
    print(f"**Proxy RSS: {int(rb)//1024} MB → {int(ra)//1024} MB** over the run "
          f"(coarse leak signal; use DURATION=10m for a soak).")
PY
echo "→ done"
