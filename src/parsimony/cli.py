"""Command-line entry point and composition root.

``parsimony serve`` wires the concrete adapters (httpx upstream, lossless compressor, SQLite
cache, redactor) to the engine and runs the ASGI app. Binding is **loopback/tailnet only** —
the configuration layer refuses a public/all-interfaces bind (fail closed).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from parsimony import __version__
from parsimony.cache import CachePolicy, NullCache, SqliteCache
from parsimony.compress import LosslessCompressor, NullCompressor
from parsimony.config import Settings
from parsimony.ports import CachePort, CompressorPort
from parsimony.proxy import create_app
from parsimony.proxy.engine import EngineConfig, ProxyEngine
from parsimony.proxy.upstream import HttpxUpstream
from parsimony.redact import Redactor

__all__ = ["build_app", "build_engine", "main"]


def build_engine(settings: Settings) -> ProxyEngine:
    """Construct the engine with concrete adapters chosen from ``settings``."""
    compressor: CompressorPort = LosslessCompressor() if settings.lossless else NullCompressor()
    cache: CachePort
    if settings.cache:
        if settings.cache_path != ":memory:":
            Path(settings.cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache = SqliteCache(path=settings.cache_path)
    else:
        cache = NullCache()
    policy = CachePolicy.from_patterns(
        settings.nocache_patterns(),
        enabled=settings.cache,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    return ProxyEngine(
        upstream=HttpxUpstream(),
        compressor=compressor,
        cache=cache,
        redactor=Redactor(),
        policy=policy,
        config=EngineConfig(
            anthropic_upstream=settings.anthropic_upstream,
            openai_upstream=settings.openai_upstream,
            cache_enabled=settings.cache,
            cache_ttl_seconds=settings.cache_ttl_seconds,
            salt=settings.salt,
        ),
    )


def build_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app from settings (defaults read from the environment)."""
    return create_app(build_engine(settings or Settings()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parsimony",
        description="Local-first, token-thrift inference proxy for agentic harnesses.",
    )
    parser.add_argument("--version", action="version", version=f"parsimony {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run the proxy (binds loopback/tailnet only).")
    serve.add_argument("--host", default=None, help="Override bind host (never 0.0.0.0).")
    serve.add_argument("--port", type=int, default=None, help="Override bind port.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested command. Returns a process exit code."""
    args = _parser().parse_args(argv)
    if args.command == "serve":
        overrides: dict[str, Any] = {}
        if args.host is not None:
            overrides["host"] = args.host
        if args.port is not None:
            overrides["port"] = args.port
        settings = Settings(**overrides)  # validates bind host (fail closed)
        uvicorn.run(
            build_app(settings),
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
    return 0
