#!/usr/bin/env python3
"""A tiny provider stand-in for fuzzing parcus in isolation.

The fuzzer fires malformed requests at `parcus serve`; parcus forwards the ones it can to its
configured upstream. Point that upstream here so fuzz traffic never reaches a real provider. This
server accepts ANY method/body and replies 200 with a fixed JSON — it must not itself error,
since we are testing parcus's robustness, not the upstream's.

Run:  python qa/fuzz/mock_upstream.py --port 8993
"""

from __future__ import annotations

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_BODY = b'{"id":"mock","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}'


class _Handler(BaseHTTPRequestHandler):
    """Drain any request and reply 200 with the canned body."""

    protocol_version = "HTTP/1.1"

    def _reply(self) -> None:
        length = int(self.headers.get("content-length", 0) or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(_BODY)))
        self.end_headers()
        self.wfile.write(_BODY)

    # BaseHTTPRequestHandler dispatches by these method-name-cased attributes.
    do_POST = _reply
    do_GET = _reply
    do_PUT = _reply
    do_DELETE = _reply

    def log_message(self, *args: object) -> None:
        """Silence per-request logging."""


def main() -> int:
    """Serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8993)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"fuzz mock upstream on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
