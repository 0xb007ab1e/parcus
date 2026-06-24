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
from parsimony.compress import (
    DEFAULT_FILLERS,
    ChainCompressor,
    FillerCompressor,
    LosslessCompressor,
    NullCompressor,
)
from parsimony.config import Settings
from parsimony.eval import (
    BUILTIN_RETRIEVAL_SAMPLES,
    BUILTIN_SAMPLES,
    evaluate,
    evaluate_retrieval,
    is_filler_equivalent,
    load_jsonl,
)
from parsimony.memory import GraphMemory
from parsimony.obs import (
    LoggingSink,
    MetricsSink,
    MultiSink,
    NullSink,
    SqliteMetricsSink,
    render_stats,
)
from parsimony.ports import CachePort, CompressorPort
from parsimony.proxy import create_app
from parsimony.proxy.engine import EngineConfig, ProxyEngine
from parsimony.proxy.upstream import HttpxUpstream
from parsimony.redact import Redactor

__all__ = ["build_app", "build_engine", "main"]


def build_engine(settings: Settings) -> ProxyEngine:
    """Construct the engine with concrete adapters chosen from ``settings``."""
    passes: list[CompressorPort] = []
    if settings.lossless:
        passes.append(LosslessCompressor())
    if settings.filler:
        passes.append(FillerCompressor())
    compressor: CompressorPort
    if not passes:
        compressor = NullCompressor()
    elif len(passes) == 1:
        compressor = passes[0]
    else:
        compressor = ChainCompressor(passes)
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
    metrics: MetricsSink
    if settings.metrics:
        if settings.metrics_path != ":memory:":
            Path(settings.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        metrics = MultiSink([LoggingSink(), SqliteMetricsSink(settings.metrics_path)])
    else:
        metrics = NullSink()
    memory = GraphMemory() if settings.memory else None
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
            memory_enabled=settings.memory,
            memory_inject=settings.memory_inject,
            memory_summarize=settings.memory_summarize,
            memory_keep_recent=settings.memory_keep_recent,
            memory_retrieve=settings.memory_retrieve,
            memory_summary_items=settings.memory_summary_items,
            memory_min_messages=settings.memory_min_messages,
        ),
        metrics=metrics,
        memory=memory,
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
    ev = sub.add_parser("eval", help="Measure token savings + equivalence over a dataset.")
    ev.add_argument(
        "dataset", nargs="?", default=None, help="JSONL dataset (default: built-in samples)."
    )
    ev.add_argument(
        "--filler",
        action="store_true",
        help="Evaluate the Tier-1 filler pass (lossless+filler) with the filler guardrail.",
    )
    ev.add_argument(
        "--retrieval",
        action="store_true",
        help="Run the memory retrieval-quality gate (recall) instead of compression eval.",
    )
    ev.add_argument(
        "--record",
        action="store_true",
        help="Record this eval's gate result into the metrics store (for `parsimony stats`).",
    )
    sub.add_parser("stats", help="Show aggregated per-stage reduction + accuracy from the store.")
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
    elif args.command == "eval":
        if args.retrieval:
            retrieval_report = evaluate_retrieval(BUILTIN_RETRIEVAL_SAMPLES)
            print(retrieval_report.render())
            if args.record:
                _record_eval("retrieval", retrieval_report.mean_score, retrieval_report.passed)
            return 0 if retrieval_report.passed else 1
        samples = load_jsonl(args.dataset) if args.dataset else BUILTIN_SAMPLES
        if args.filler:
            report = evaluate(
                samples,
                compressor=ChainCompressor([LosslessCompressor(), FillerCompressor()]),
                equivalence=lambda o, c: is_filler_equivalent(o, c, DEFAULT_FILLERS),
            )
        else:
            report = evaluate(samples)
        print(report.render())
        if args.record:
            # Equivalence is binary: accuracy 1.0 when the gate held, else 0.0.
            _record_eval(
                "filler" if args.filler else "lossless",
                1.0 if report.passed else 0.0,
                report.passed,
            )
        return 0 if report.passed else 1
    elif args.command == "stats":
        store = SqliteMetricsSink(Settings().metrics_path)
        try:
            print(render_stats(store.snapshot()))
        finally:
            store.close()
    return 0


def _record_eval(kind: str, score: float, passed: bool) -> None:
    """Persist an eval-gate result to the metrics store (for `parsimony stats`)."""
    store = SqliteMetricsSink(Settings().metrics_path)
    try:
        store.record_eval(kind, score, passed)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
