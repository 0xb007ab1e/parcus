#!/usr/bin/env python3
"""Atheris (libFuzzer) coverage-guided fuzz target for parcus's untrusted-input parsers.

The request edge parses bytes from the network into the canonical model. Those parsers must
**fail open** — return ``None``/``False`` or a value, never raise — because the engine calls some
of them (notably ``dialects.parse``) *outside* a try block, so an unhandled parser exception would
surface as a 5xx and break the harness. This target feeds fuzzer-mutated input to each parser
boundary and lets any exception propagate, so libFuzzer records it as a crash (= a fail-open gap).

Complements the black-box HTTP fuzzing in ``run.sh`` (schemathesis): this drives the *functions*
directly, coverage-guided, reaching edge cases black-box traffic rarely hits.

Run:  qa/fuzz/run_atheris.sh            (or: python qa/fuzz/fuzz_parsers.py -max_total_time=30)
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import json

    from parcus.model import Dialect
    from parcus.proxy.app import _is_stream
    from parcus.proxy.dialects import detect, parse
    from parcus.redact import Redactor
    from parcus.tenant import derive_tenant

_REDACTOR = Redactor()


def _exercise(data: bytes) -> None:
    """Feed one fuzz input to every untrusted-input parser; none may raise."""
    fdp = atheris.FuzzedDataProvider(data)
    text = fdp.ConsumeUnicodeNoSurrogates(max(1, fdp.remaining_bytes() // 2))
    body = fdp.ConsumeBytes(fdp.remaining_bytes())

    detect(text)  # path → dialect (pure string parse)
    _is_stream(body)  # body → "is this a streaming request?" (must tolerate any bytes)

    # json.loads legitimately raises on non-JSON — that is the harness's concern, mirroring
    # engine._canonicalize. But parse() runs un-guarded in the engine, so it must NOT raise.
    try:
        decoded = json.loads(body or b"null")
    except (ValueError, UnicodeDecodeError):
        decoded = None
    if isinstance(decoded, dict):
        parse(Dialect.ANTHROPIC, decoded)
        parse(Dialect.OPENAI, decoded)

    derive_tenant([("x-api-key", text), ("authorization", text)])  # server-side tenant id
    _REDACTOR.has_secret(text)  # secret-scan regex robustness (no crash / catastrophic backtrack)


def test_one_input(data: bytes) -> None:
    """Entry point libFuzzer calls once per mutated input."""
    _exercise(data)


def main() -> None:
    """Wire the target into Atheris and start fuzzing."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
