// k6 load script for parcus. Measures latency/throughput against a target base URL.
//
// Driven by env vars (set by run.sh, or pass with `-e`):
//   TARGET   base URL to hit (the mock for a baseline, or parcus for the proxied path)
//   MODE     "miss" = a unique body per request (always forwarded + cached)
//            "hit"  = a constant body (after warmup, served from parcus's cache, no upstream hop)
//   VUS      virtual users (default 10)
//   DURATION test duration (default 10s; set e.g. 10m for a soak/leak run)
//
// The only hard threshold is correctness (≈0 failed requests); latency is reported, not gated,
// so a run always yields numbers to compare.
import http from "k6/http";
import { check } from "k6";

const TARGET = __ENV.TARGET || "http://127.0.0.1:8787";
const MODE = __ENV.MODE || "miss";

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || "10s",
  thresholds: {
    http_req_failed: ["rate<0.01"], // requests must succeed; parcus never breaks the call
  },
};

export default function () {
  // "hit": every VU sends the SAME body so parcus replays it from cache after the first call.
  // "miss": a unique body each iteration so every request is forwarded upstream and stored.
  const tag = MODE === "hit" ? "constant" : `${__VU}-${__ITER}`;
  const body = JSON.stringify({
    model: "claude-sonnet-4-6",
    messages: [{ role: "user", content: `please just briefly summarize task ${tag}` }],
  });
  const res = http.post(`${TARGET}/v1/messages`, body, {
    headers: { "Content-Type": "application/json", "x-api-key": "loadtest-key" },
  });
  check(res, { "status is 200": (r) => r.status === 200 });
}
