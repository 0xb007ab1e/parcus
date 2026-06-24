"""Tests for the CLI / composition root (without launching a server)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from parsimony import cli
from parsimony.compress import ChainCompressor, NullCompressor
from parsimony.config import Settings


def test_build_app_returns_fastapi_instance() -> None:
    app = cli.build_app(Settings(_env_file=None, cache=False))
    assert isinstance(app, FastAPI)


def test_serve_invokes_uvicorn_with_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: object, host: str, port: int, log_level: str) -> None:
        captured.update(host=host, port=port, log_level=log_level)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "build_app", lambda _s: object())

    rc = cli.main(["serve", "--host", "127.0.0.1", "--port", "9991"])
    assert rc == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9991


def test_eval_command_runs_builtin_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["eval"])
    assert rc == 0  # built-in corpus is lossless → passes the equivalence gate
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "PASS" in out


def test_eval_filler_command_runs_builtin_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["eval", "--filler"])
    assert rc == 0  # built-in corpus passes the filler guardrail
    assert "TOTAL" in capsys.readouterr().out


def test_build_engine_chains_both_passes() -> None:
    engine = cli.build_engine(Settings(_env_file=None, cache=False, lossless=True, filler=True))
    assert isinstance(engine._compressor, ChainCompressor)


def test_build_engine_uses_null_when_no_passes() -> None:
    engine = cli.build_engine(Settings(_env_file=None, cache=False, lossless=False, filler=False))
    assert isinstance(engine._compressor, NullCompressor)


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_serve_rejects_public_bind() -> None:
    with pytest.raises(ValidationError):
        cli.main(["serve", "--host", "0.0.0.0"])  # noqa: S104
