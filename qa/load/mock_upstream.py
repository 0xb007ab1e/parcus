#!/usr/bin/env python3
"""A minimal, fast mock provider upstream for load testing parcus in isolation.

Load tests must measure *parcus's own* per-request overhead, not a real provider's latency or
rate limits. This server stands in for the provider: it consumes the request body and returns a
small canned JSON response as fast as possible, so the latency delta between "client → mock" and
"client → parcus → mock" is parcus's processing cost (parse → compress → cache decision →
re-serialize → forward).

Threaded so concurrent virtual users don't serialize on the handler. No network, no secrets.

Run:  python qa/load/mock_upstream.py --port 89 91
"""

from __future__ import annotations

import argparse
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_BODY = b'{"id":"mock","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}'


class _Handler(BaseHTTPRequestHandler):
    """Respond to any POST with a fixed JSON body; ignore everything else cheaply."""

    protocol_version = "HTTP/1.1"  # keep-alive so connection setup isn't measured each request

    def setup(self) -> None:
        """Disable Nagle's algorithm on each connection.

        Otherwise small replies incur a ~40 ms delayed-ACK stall that would swamp the latency
        the harness is trying to measure.
        """
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_POST(self) -> None:  # BaseHTTPRequestHandler dispatches by method-name casing
        """Drain the request body and reply with the canned response."""
        length = int(self.headers.get("content-length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(_BODY)))
        self.end_headers()
        self.wfile.write(_BODY)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging (it would dominate the measured overhead)."""


def main() -> int:
    """Parse ``--host``/``--port`` and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8991)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"mock upstream on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
