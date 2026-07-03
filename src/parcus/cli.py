"""Command-line entry point and composition root.

``parcus serve`` wires the concrete adapters (httpx upstream, lossless compressor, SQLite
cache, redactor) to the engine and runs the ASGI app. Binding is **loopback/tailnet only** —
the configuration layer refuses a public/all-interfaces bind (fail closed).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from parcus import __version__
from parcus.cache import (
    CachePolicy,
    NullCache,
    SimilarityCache,
    SqliteCache,
    SqliteSimilarityStore,
)
from parcus.compress import (
    AGGRESSIVE_FILLERS,
    DEFAULT_FILLERS,
    ChainCompressor,
    DedupCompressor,
    FillerCompressor,
    LearnedCompressor,
    LLMLinguaReducer,
    LosslessCompressor,
    NullCompressor,
    ToolResultElider,
)
from parcus.config import Settings
from parcus.eval import (
    BUILTIN_DEDUP_SAMPLES,
    BUILTIN_ELISION_SAMPLES,
    BUILTIN_JUDGED_SAMPLES,
    BUILTIN_RETRIEVAL_SAMPLES,
    BUILTIN_SAMPLES,
    BUILTIN_SIMILARITY_SAMPLES,
    KeywordRecallJudge,
    evaluate,
    evaluate_judged,
    evaluate_judged_dedup,
    evaluate_judged_elision,
    evaluate_retrieval,
    evaluate_similarity,
    is_filler_equivalent,
    load_jsonl,
)
from parcus.memory import (
    EmbedderPort,
    GraphMemory,
    HashingEmbedder,
    MemoryProvider,
    PerTenantMemoryProvider,
    SentenceTransformerEmbedder,
    SharedMemoryProvider,
)
from parcus.obs import (
    LoggingSink,
    MetricsSink,
    MultiSink,
    NullSink,
    SqliteMetricsSink,
    render_stats,
)
from parcus.ports import CachePort, CompressorPort
from parcus.proxy import create_app
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import HttpxUpstream
from parcus.quota import RateLimiter
from parcus.redact import Redactor
from parcus.tenant import derive_tenant

__all__ = ["build_app", "build_engine", "main"]

# Default keep-ratio sweep for `eval --learned --sweep` (bare flag): coarse-to-fine, all in (0, 1].
_DEFAULT_SWEEP_RATIOS = "0.3,0.5,0.7,0.9"


def _build_metrics(settings: Settings) -> tuple[MetricsSink, SqliteMetricsSink | None]:
    """Return the engine's metrics sink and the persistent store (if metrics are enabled).

    The store is returned separately so it can also back the stats endpoint.
    """
    if not settings.metrics:
        return NullSink(), None
    if settings.metrics_path != ":memory:":
        Path(settings.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteMetricsSink(settings.metrics_path)
    return MultiSink([LoggingSink(), store]), store


def build_engine(settings: Settings, *, metrics: MetricsSink | None = None) -> ProxyEngine:
    """Construct the engine with concrete adapters chosen from ``settings``."""
    rate = settings.invariant_sample_rate
    passes: list[CompressorPort] = []
    if settings.lossless:
        passes.append(LosslessCompressor(verify_sample=rate))
    if settings.filler:
        fillers = AGGRESSIVE_FILLERS if settings.filler_aggressive else DEFAULT_FILLERS
        passes.append(FillerCompressor(fillers=fillers, verify_sample=rate))
    if settings.learned:
        # Local LLMLingua reducer; model loads lazily on first use (the 'learned' extra). Last
        # in the chain — operate on already-losslessly/filler-trimmed prose. The LLMLingua-2
        # backend is opt-in (higher fidelity; validate offline first).
        reducer = LLMLinguaReducer(
            model_name=settings.learned_model or None,
            use_llmlingua2=settings.learned_llmlingua2,
        )
        passes.append(LearnedCompressor(reducer, keep_ratio=settings.learned_ratio))
    if settings.elide_tool_results:
        # Lossy: stub stale tool_result payloads in structured turns (needs parse_structured).
        passes.append(ToolResultElider(keep_recent=settings.elide_keep_recent))
    if settings.dedup:
        # Lossy: collapse later byte-identical copies of a large block to a reference.
        passes.append(DedupCompressor(min_chars=settings.dedup_min_chars))
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
        cache = _maybe_encrypt(cache, settings)
    else:
        cache = NullCache()
    policy = CachePolicy.from_patterns(
        settings.nocache_patterns(),
        enabled=settings.cache,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    metrics_sink = metrics if metrics is not None else _build_metrics(settings)[0]
    memory_provider = _build_memory_provider(settings)
    rate_limit = settings.rate_limit()
    rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None
    similarity = _build_similarity(settings)
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
            cache_inject=settings.cache_inject,
            cache_inject_repeat_aware=settings.cache_inject_repeat_aware,
            parse_structured=settings.parse_structured,
            memory_enabled=settings.memory,
            memory_inject=settings.memory_inject,
            memory_summarize=settings.memory_summarize,
            memory_keep_recent=settings.memory_keep_recent,
            memory_retrieve=settings.memory_retrieve,
            memory_summary_items=settings.memory_summary_items,
            memory_min_messages=settings.memory_min_messages,
            multi_tenant=settings.multi_tenant,
            allowed_tenants=settings.allowed_tenant_set(),
        ),
        metrics=metrics_sink,
        memory_provider=memory_provider,
        rate_limiter=rate_limiter,
        similarity=similarity,
    )


def _maybe_encrypt(cache: CachePort, settings: Settings) -> CachePort:
    """Wrap the cache in at-rest encryption when enabled (lazy import keeps crypto optional).

    The key is validated by settings; it is present here because enabling encryption without a
    valid key already failed closed at settings construction.
    """
    if not settings.cache_encryption:
        return cache
    from parcus.cache.encryption import CacheCipher, EncryptedCache, TenantCipherProvider

    key = settings.cache_encryption_key_bytes()
    if key is None:  # defensive: settings validation already guarantees a key here (fail closed)
        raise RuntimeError("cache encryption enabled without a valid key")
    previous = settings.cache_encryption_previous_key_bytes()
    if settings.multi_tenant:
        # Per-tenant DEKs derived from the master key, with crypto-shredding by withheld key.
        provider = TenantCipherProvider(
            key, previous_master_keys=previous, shredded=settings.cache_shredded_tenant_set()
        )
        return EncryptedCache(cache, provider=provider)
    # Single-tenant: one cipher straight from the master key (unchanged behaviour).
    return EncryptedCache(cache, CacheCipher(key, previous_keys=previous))


def _build_similarity(settings: Settings) -> SimilarityCache | None:
    """Build the opt-in semantic cache (local embedder), or ``None`` when disabled.

    When ``similarity_persist`` is set, back it with a confidential sidecar snapshot so
    near-duplicate hits survive a restart (the index still operates in memory).
    """
    if not settings.similarity_cache:
        return None
    embedder = _embedder(settings.similarity_embedder) or HashingEmbedder()
    store: SqliteSimilarityStore | None = None
    if settings.similarity_persist:
        if settings.similarity_path != ":memory:":
            Path(settings.similarity_path).parent.mkdir(parents=True, exist_ok=True)
        store = SqliteSimilarityStore(
            settings.similarity_path, max_entries=settings.similarity_max_entries
        )
    return SimilarityCache(
        embedder,
        threshold=settings.similarity_threshold,
        max_entries=settings.similarity_max_entries,
        store=store,
    )


def _build_memory_provider(settings: Settings) -> MemoryProvider | None:
    """Build the memory provider: per-tenant graphs in hosted mode, one shared graph locally."""
    if not settings.memory:
        return None
    if settings.multi_tenant:
        # A fresh graph per tenant id so context never crosses tenants (isolation boundary).
        return PerTenantMemoryProvider(lambda: GraphMemory())
    return SharedMemoryProvider(GraphMemory())


def build_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app from settings (defaults read from the environment)."""
    settings = settings or Settings()
    metrics_sink, store = _build_metrics(settings)
    engine = build_engine(settings, metrics=metrics_sink)
    return create_app(engine, stats_source=store)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parcus",
        description="Local-first, token-thrift inference proxy for agentic harnesses.",
    )
    parser.add_argument("--version", action="version", version=f"parcus {__version__}")
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
        "--aggressive",
        action="store_true",
        help="With --filler, use the larger AGGRESSIVE_FILLERS set instead of the default.",
    )
    ev.add_argument(
        "--judged",
        action="store_true",
        help="Gate the selected tier by answer-preservation (recall judge) over the built-in "
        "corpus, not just the structural invariant — e.g. validate an aggressive filler set.",
    )
    ev.add_argument(
        "--elision",
        action="store_true",
        help="Answer-preservation gate for tool-result elision (M1d slice 3): confirm stale "
        "tool_result payloads are dropped while recent answer-relevant content survives.",
    )
    ev.add_argument(
        "--dedup",
        action="store_true",
        help="Answer-preservation gate for cross-turn dedup (Tier-2): confirm a repeated block is "
        "collapsed while the answer-relevant content in the kept first copy survives.",
    )
    ev.add_argument(
        "--learned",
        action="store_true",
        help="Answer-preservation gate for the Tier-2 learned compressor; skips (CI-safe, exit 0) "
        "when the local model / 'learned' extra is unavailable.",
    )
    ev.add_argument(
        "--sweep",
        nargs="?",
        const=_DEFAULT_SWEEP_RATIOS,
        default=None,
        metavar="RATIOS",
        help="With --learned: sweep the gate over a comma-separated list of keep-ratios "
        f"(default '{_DEFAULT_SWEEP_RATIOS}' when the flag is bare) and report the lowest ratio "
        "that still clears the answer-preservation bar. Exit 0 if any ratio passes.",
    )
    ev.add_argument(
        "--retrieval",
        action="store_true",
        help="Run the memory retrieval-quality gate (recall) instead of compression eval.",
    )
    ev.add_argument(
        "--similarity",
        action="store_true",
        help="Run the semantic-cache precision gate (no false hits) instead of compression eval.",
    )
    ev.add_argument(
        "--threshold",
        type=float,
        default=0.97,
        help="Cosine threshold for --similarity (default 0.97).",
    )
    ev.add_argument(
        "--embedder",
        choices=["lexical", "hashing", "local"],
        default="lexical",
        help="Embedder for --retrieval/--similarity: lexical (default), hashing, or local "
        "(sentence-transformers; requires the 'embeddings' extra).",
    )
    ev.add_argument(
        "--record",
        action="store_true",
        help="Record this eval's gate result into the metrics store (for `parcus stats`).",
    )
    sub.add_parser("stats", help="Show aggregated per-stage reduction + accuracy from the store.")
    sub.add_parser(
        "tenant-id",
        help="Print the tenant id for a credential (read from PARCUS_TENANT_CREDENTIAL or "
        "stdin, never argv) to build PARCUS_ALLOWED_TENANTS.",
    )
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
            retrieval_report = evaluate_retrieval(
                BUILTIN_RETRIEVAL_SAMPLES, embedder=_embedder(args.embedder)
            )
            print(retrieval_report.render())
            if args.record:
                _record_eval("retrieval", retrieval_report.mean_score, retrieval_report.passed)
            return 0 if retrieval_report.passed else 1
        if args.similarity:
            sim_report = evaluate_similarity(
                BUILTIN_SIMILARITY_SAMPLES,
                threshold=args.threshold,
                embedder=_embedder(args.embedder),
            )
            print(sim_report.render())
            if args.record:
                _record_eval("similarity", sim_report.precision, sim_report.passed)
            return 0 if sim_report.passed else 1
        if args.dedup:
            return _eval_dedup(args.record)
        if args.elision:
            return _eval_elision(args.record)
        if args.learned:
            return _eval_learned(args.record, sweep=args.sweep)
        if args.judged:
            return _eval_judged(args.filler, args.aggressive, args.record)
        samples = load_jsonl(args.dataset) if args.dataset else BUILTIN_SAMPLES
        if args.filler:
            fillers = AGGRESSIVE_FILLERS if args.aggressive else DEFAULT_FILLERS
            report = evaluate(
                samples,
                compressor=ChainCompressor(
                    [LosslessCompressor(), FillerCompressor(fillers=fillers)]
                ),
                equivalence=lambda o, c: is_filler_equivalent(o, c, fillers),
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
    elif args.command == "tenant-id":
        return _print_tenant_id()
    return 0


def _print_tenant_id() -> int:
    """Print the tenant id for a credential, using the install salt. Returns an exit code.

    The credential is read from ``PARCUS_TENANT_CREDENTIAL`` or, if unset, stdin — **never**
    from argv (which would leak it into shell history / the process table; see lang-shell,
    workflow-secrets). The salt comes from settings so the id matches what the running proxy
    derives.
    """
    settings = Settings()
    credential = os.environ.get("PARCUS_TENANT_CREDENTIAL") or sys.stdin.readline().strip()
    if not credential:
        print(
            "no credential provided (set PARCUS_TENANT_CREDENTIAL or pipe it on stdin)",
            file=sys.stderr,
        )
        return 1
    print(derive_tenant([("x-api-key", credential)], salt=settings.salt))
    return 0


def _embedder(name: str) -> EmbedderPort | None:
    """Map the --embedder choice to an embedder (None = lexical retrieval)."""
    if name == "hashing":
        return HashingEmbedder()
    if name == "local":
        return SentenceTransformerEmbedder()
    return None


def _record_eval(kind: str, score: float, passed: bool) -> None:
    """Persist an eval-gate result to the metrics store (for `parcus stats`)."""
    store = SqliteMetricsSink(Settings().metrics_path)
    try:
        store.record_eval(kind, score, passed)
    finally:
        store.close()


def _eval_judged(filler: bool, aggressive: bool, record: bool) -> int:
    """Answer-preservation gate over the (lossless [+ filler]) tier — CI-safe, model-free.

    Used to validate an aggressive filler set: confirm the compressed prompts still preserve the
    built-in corpus's required content phrases.
    """
    fillers = AGGRESSIVE_FILLERS if aggressive else DEFAULT_FILLERS
    passes: list[CompressorPort] = [LosslessCompressor()]
    if filler:
        passes.append(FillerCompressor(fillers=fillers))
    report = evaluate_judged(BUILTIN_JUDGED_SAMPLES, ChainCompressor(passes), KeywordRecallJudge())
    print(report.render())
    if record:
        kind = "judged-aggressive" if aggressive else "judged"
        _record_eval(kind, report.mean_score, report.passed)
    return 0 if report.passed else 1


def _eval_dedup(record: bool) -> int:
    """Answer-preservation gate for cross-turn dedup — CI-safe, model-free.

    Confirms that collapsing a repeated block to a reference keeps the answer-relevant content (in
    the kept first copy) on the built-in corpus, and that a duplicate was actually deduplicated.
    """
    report = evaluate_judged_dedup(BUILTIN_DEDUP_SAMPLES, DedupCompressor(), KeywordRecallJudge())
    print(report.render())
    if record:
        _record_eval("dedup", report.mean_score, report.passed)
    return 0 if report.passed else 1


def _eval_elision(record: bool) -> int:
    """Answer-preservation gate for tool-result elision — CI-safe, model-free.

    Confirms that eliding stale ``tool_result`` payloads (keep_recent=4, the shipping default)
    drops the stale content while the recent answer-relevant facts survive on the built-in corpus.
    """
    report = evaluate_judged_elision(
        BUILTIN_ELISION_SAMPLES, ToolResultElider(keep_recent=4), KeywordRecallJudge()
    )
    print(report.render())
    if record:
        _record_eval("elision", report.mean_score, report.passed)
    return 0 if report.passed else 1


def _parse_sweep_ratios(spec: str) -> list[float]:
    """Parse a comma-separated keep-ratio sweep spec into validated ratios in (0, 1].

    Raises ``ValueError`` on non-numeric entries, out-of-range ratios, or an empty spec — so a bad
    ``--sweep`` fails fast (before any model load) with a clear message.
    """
    ratios: list[float] = []
    for part in spec.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"keep-ratio must be in (0.0, 1.0], got {value}")
        ratios.append(value)
    if not ratios:
        raise ValueError("no keep-ratios given")
    return ratios


def _learned_chain(reducer: LLMLinguaReducer, keep_ratio: float) -> ChainCompressor:
    """The shipping learned chain (lossless → aggressive filler → learned) at ``keep_ratio``."""
    return ChainCompressor(
        [
            LosslessCompressor(),
            FillerCompressor(fillers=AGGRESSIVE_FILLERS),
            LearnedCompressor(reducer, keep_ratio=keep_ratio),
        ]
    )


def _eval_learned(record: bool, sweep: str | None = None) -> int:
    """Answer-preservation gate for the Tier-2 learned compressor.

    Needs a local LLMLingua model; when it is unavailable (CI / no 'learned' extra) this skips
    with exit 0 rather than failing the gate. The model path is exercised offline. The gate logic
    itself (``evaluate_judged`` + the recall judge) is covered in CI via a fake reducer.

    With ``sweep`` set (a comma-separated keep-ratio spec) it runs the gate at each ratio and
    reports the lowest ratio that still clears the bar (exit 0 if any passes) — the tool for
    finding a viable setting for a more aggressive backend (e.g. LLMLingua-2).
    """
    try:
        ratios = _parse_sweep_ratios(sweep) if sweep is not None else [0.5]
    except ValueError as exc:
        print(f"parcus eval --learned: invalid --sweep: {exc}")
        return 2
    use_llmlingua2 = os.environ.get("PARCUS_LEARNED_LLMLINGUA2", "").lower() in ("1", "true", "yes")
    reducer = LLMLinguaReducer(
        model_name=os.environ.get("PARCUS_LEARNED_MODEL") or None,
        use_llmlingua2=use_llmlingua2,
    )
    try:
        reducer.reduce("a short probe prompt", keep_ratio=0.5)
    except Exception:
        print(f"parcus eval --learned: skipped ({reducer.model_name} unavailable)")
        return 0
    return _run_learned_gate(reducer, ratios, record, is_sweep=sweep is not None)


def _run_learned_gate(  # pragma: no cover - only when a local model is present
    reducer: LLMLinguaReducer, ratios: list[float], record: bool, *, is_sweep: bool
) -> int:
    """Run the judged gate at each keep-ratio; single-report for one ratio, summary for a sweep."""
    judge = KeywordRecallJudge()
    reports = [
        (r, evaluate_judged(BUILTIN_JUDGED_SAMPLES, _learned_chain(reducer, r), judge))
        for r in ratios
    ]
    if not is_sweep:
        _ratio, report = reports[0]
        print(report.render())
        if record:
            _record_eval("learned", report.mean_score, report.passed)
        return 0 if report.passed else 1

    lines = [f"{'keep_ratio':>10}  {'mean recall':>11}  {'verdict':>7}", "-" * 32]
    for r, rep in reports:
        lines.append(
            f"{r:>10.2f}  {rep.mean_score:>11.0%}  {('PASS' if rep.passed else 'FAIL'):>7}"
        )
    print("\n".join(lines))

    passing = [(r, rep) for r, rep in reports if rep.passed]
    if passing:
        best_r, best_rep = min(passing, key=lambda x: x[0])  # lowest ratio = most compression
        print(f"best passing keep_ratio: {best_r:.2f} ({best_rep.mean_score:.0%} recall)")
        if record:
            _record_eval("learned-sweep", best_rep.mean_score, True)
        return 0
    print("no keep_ratio cleared the answer-preservation bar")
    if record:
        best_rep = max((rep for _r, rep in reports), key=lambda rp: rp.mean_score)
        _record_eval("learned-sweep", best_rep.mean_score, False)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
