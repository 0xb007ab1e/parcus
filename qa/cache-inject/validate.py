#!/usr/bin/env python3
"""Validate M1b prompt-cache injection against a live explicit-breakpoint provider (Anthropic).

parcus (M1b) can inject a provider cache breakpoint (Anthropic ``cache_control``) onto a large
stable request prefix so the provider serves it from its prompt cache on the next turn — the
dominant cost lever for tool/history-heavy harnesses. This harness measures that, using the
provider's **own** ``cache_read_input_tokens`` (ground truth, captured by parcus into
``upstream_usage``):

* it sends the same large-prefix request through parcus **twice** per condition, and
* compares condition ``inject off`` (baseline) vs ``inject on``.

Expected: with injection **on**, the second turn reads the stable prefix from the provider's
cache (``cache_read`` ≈ prefix tokens); with it **off**, nothing is cached (``cache_read`` = 0).

Usage:
  # Live run — needs an Anthropic key + network (your key; nothing is printed):
  ANTHROPIC_API_KEY=... .venv/bin/python qa/cache-inject/validate.py [--model M] [--prefix-tokens N]

  # Offline harness check — no key, no network — proves parcus injects cache_control only when
  # enabled and that this harness reads usage correctly (against a fake Anthropic upstream):
  .venv/bin/python qa/cache-inject/validate.py --self-test

Not a project dependency and not run in CI (``qa/`` is out-of-band). No secrets are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from parcus.cache import CachePolicy, NullCache
from parcus.compress import NullCompressor
from parcus.model import ProviderUsage
from parcus.proxy.engine import EngineConfig, ProxyEngine
from parcus.proxy.upstream import UpstreamRequest, UpstreamResponse
from parcus.redact import Redactor
from parcus.tokenize import default_tokenizer

_ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
_PATH = "/v1/messages"
_DEFAULT_MODEL = os.environ.get("PARCUS_VALIDATE_MODEL", "claude-haiku-4-5")
_STABLE_PARAGRAPH = (
    "The parcus proxy sits transparently between an agentic harness and the model provider, "
    "compressing the outbound request and caching responses to reduce tokens spent per turn "
    "while preserving the meaning the model needs to act on the task correctly. "
)


# --- request construction --------------------------------------------------------------------


def build_prefix(target_tokens: int, model: str | None) -> str:
    """Return stable filler whose parcus-tokenizer count is at least ``target_tokens``."""
    tok = default_tokenizer()
    per = tok.count(_STABLE_PARAGRAPH, model) or 1
    text = _STABLE_PARAGRAPH * max(1, target_tokens // per + 2)
    while tok.count(text, model) < target_tokens:
        text += _STABLE_PARAGRAPH
    return text


def build_body(model: str, prefix: str, question: str, *, max_tokens: int = 16) -> bytes:
    """Anthropic Messages body: big stable ``system`` + a stable turn + a volatile question.

    Three messages → the M1b boundary protects ``system`` + the first two turns, so the
    breakpoint lands on the last stable turn and the final question stays uncached (volatile).
    """
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": prefix,
        "messages": [
            {"role": "user", "content": "Here is the project context above. Acknowledge it."},
            {"role": "assistant", "content": "Acknowledged. Ready for your question."},
            {"role": "user", "content": question},
        ],
    }
    return json.dumps(body).encode()


# --- engine + condition runner ---------------------------------------------------------------


def build_engine(*, cache_inject: bool, upstream: Any) -> ProxyEngine:
    """A minimal engine that isolates the injection effect (no compression, no parcus cache)."""
    return ProxyEngine(
        upstream=upstream,
        compressor=NullCompressor(),  # isolate injection: the only variable is cache_control
        cache=NullCache(),
        redactor=Redactor(),
        policy=CachePolicy(),
        config=EngineConfig(
            anthropic_upstream=_ANTHROPIC_UPSTREAM,
            openai_upstream="",
            cache_enabled=False,  # observe the PROVIDER's cache, not parcus's own
            cache_inject=cache_inject,
        ),
    )


async def run_condition(
    upstream: Any,
    *,
    cache_inject: bool,
    headers: list[tuple[str, str]],
    body: bytes,
    turns: int = 2,
) -> list[ProviderUsage | None]:
    """Send ``body`` through parcus ``turns`` times; return the provider usage captured per turn."""
    engine = build_engine(cache_inject=cache_inject, upstream=upstream)
    usages: list[ProviderUsage | None] = []
    for _ in range(turns):
        result = await engine.handle("POST", _PATH, headers, body)
        usage = result.meta.get("upstream_usage")
        usages.append(usage if isinstance(usage, ProviderUsage) else None)
    return usages


# --- reporting -------------------------------------------------------------------------------


def _fmt(usages: list[ProviderUsage | None]) -> str:
    cells = []
    for u in usages:
        if u is None:
            cells.append("in=? cw=? cr=?")
        else:
            cells.append(f"in={u.input_tokens} cw={u.cache_write_tokens} cr={u.cache_read_tokens}")
    return "  |  ".join(f"turn{i + 1}: {c}" for i, c in enumerate(cells))


def format_report(results: dict[str, list[ProviderUsage | None]], *, prefix_tokens: int) -> str:
    """Render the baseline-vs-inject matrix and the headline cache-read delta."""
    lines = [
        f"Stable prefix (parcus tiktoken estimate): ~{prefix_tokens} tokens",
        "",
        "condition               (in=input  cw=cache_write  cr=cache_read, provider-billed)",
    ]
    for label, usages in results.items():
        lines.append(f"  {label:<22} {_fmt(usages)}")
    base = results.get("baseline (inject off)")
    inj = results.get("inject on")
    if base and inj and len(base) >= 2 and len(inj) >= 2 and base[1] and inj[1]:
        base_read = base[1].cache_read_tokens
        inj_read = inj[1].cache_read_tokens
        lines += [
            "",
            f"Turn-2 cache_read — baseline: {base_read}   inject: {inj_read}   "
            f"delta: {inj_read - base_read}",
            (
                "→ injection served the stable prefix from the provider cache."
                if inj_read > base_read
                else "→ no cache-read gain observed (check prefix size / model min / provider)."
            ),
        ]
    return "\n".join(lines)


# --- offline self-test (fake Anthropic; no key, no network) ----------------------------------


def _has_cache_control(body: dict[str, Any]) -> bool:
    """True if any message content is a block list carrying a cache_control marker."""
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


class _FakeAnthropic:
    """Offline Anthropic stand-in: caches a prefix only when the request carries cache_control.

    Mirrors the provider closely enough to prove two things end-to-end without a network call:
    that parcus actually injects ``cache_control`` (the fake inspects the forwarded body), and
    that this harness reads ``upstream_usage`` correctly.
    """

    def __init__(self) -> None:
        self._written: set[str] = set()

    async def send(self, request: UpstreamRequest) -> UpstreamResponse:
        body = json.loads(request.content)
        prefix_key = json.dumps(
            [body.get("system"), body.get("messages", [])[:-1]], sort_keys=True, default=str
        )
        input_tokens = 5000
        if _has_cache_control(body):
            if prefix_key in self._written:
                cache_read, cache_write = input_tokens, 0
            else:
                self._written.add(prefix_key)
                cache_read, cache_write = 0, input_tokens
        else:
            cache_read, cache_write = 0, 0
        payload = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 5,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
        }
        return UpstreamResponse(
            200, (("content-type", "application/json"),), json.dumps(payload).encode()
        )


def self_test() -> int:
    """Prove the injection path + harness offline. Returns a process exit code."""
    tok = default_tokenizer()
    prefix = build_prefix(4200, "m")  # clear the 4096-token Anthropic injection gate
    body = build_body("m", prefix, "In one word, are you ready?")
    headers = [("x-api-key", "test-not-a-real-key"), ("anthropic-version", "2023-06-01")]

    async def run() -> tuple[list[ProviderUsage | None], list[ProviderUsage | None]]:
        off = await run_condition(
            _FakeAnthropic(), cache_inject=False, headers=headers, body=body
        )
        on = await run_condition(_FakeAnthropic(), cache_inject=True, headers=headers, body=body)
        return off, on

    off, on = asyncio.run(run())
    assert off[1] is not None and off[1].cache_read_tokens == 0, "inject-off must not cache-read"
    assert on[0] is not None and on[0].cache_write_tokens > 0, "inject-on turn1 must write cache"
    assert on[1] is not None and on[1].cache_read_tokens > 0, "inject-on turn2 must read cache"
    print("self-test OK — parcus injects cache_control only when enabled; harness reads usage.\n")
    print(
        format_report(
            {"baseline (inject off)": off, "inject on": on},
            prefix_tokens=tok.count(prefix, "m"),
        )
    )
    return 0


# --- live run ---------------------------------------------------------------------------------


async def _live(model: str, prefix_tokens: int) -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY not set — cannot run the live validation.", file=sys.stderr)
        return 2
    from parcus.proxy.upstream import HttpxUpstream

    tok = default_tokenizer()
    prefix = build_prefix(prefix_tokens, model)
    body = build_body(model, prefix, "In one word, are you ready?")
    headers = [
        ("x-api-key", key),
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ]
    results: dict[str, list[ProviderUsage | None]] = {}
    for label, inject in (("baseline (inject off)", False), ("inject on", True)):
        results[label] = await run_condition(
            HttpxUpstream(), cache_inject=inject, headers=headers, body=body
        )
    print(format_report(results, prefix_tokens=tok.count(prefix, model)))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate M1b prompt-cache injection.")
    parser.add_argument(
        "--self-test", action="store_true", help="offline harness check (no key, no network)"
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Anthropic model id (live run)")
    parser.add_argument(
        "--prefix-tokens", type=int, default=5000, help="target stable-prefix token count"
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    return asyncio.run(_live(args.model, args.prefix_tokens))


if __name__ == "__main__":
    raise SystemExit(main())
