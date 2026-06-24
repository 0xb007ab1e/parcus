"""Tests for the CLI / composition root (without launching a server)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from parsimony import cli
from parsimony.compress import ChainCompressor, NullCompressor
from parsimony.config import Settings


def test_build_app_returns_fastapi_instance() -> None:
    app = cli.build_app(Settings(_env_file=None, cache=False, metrics=False))
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


def test_eval_retrieval_gate_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["eval", "--retrieval"])
    assert rc == 0  # built-in retrieval samples pass the recall gate
    assert "mean_recall" in capsys.readouterr().out


def test_build_engine_chains_both_passes() -> None:
    engine = cli.build_engine(
        Settings(_env_file=None, cache=False, metrics=False, lossless=True, filler=True)
    )
    assert isinstance(engine._compressor, ChainCompressor)


def test_build_engine_uses_null_when_no_passes() -> None:
    engine = cli.build_engine(
        Settings(_env_file=None, cache=False, metrics=False, lossless=False, filler=False)
    )
    assert isinstance(engine._compressor, NullCompressor)


def test_stats_command_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PARSIMONY_METRICS_PATH", str(tmp_path / "m.sqlite"))
    assert cli.main(["stats"]) == 0  # empty store still renders
    assert "requests=" in capsys.readouterr().out


def test_eval_record_then_stats_shows_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PARSIMONY_METRICS_PATH", str(tmp_path / "m.sqlite"))
    assert cli.main(["eval", "--record"]) == 0
    capsys.readouterr()  # discard eval output
    cli.main(["stats"])
    assert "lossless" in capsys.readouterr().out


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_serve_rejects_public_bind() -> None:
    with pytest.raises(ValidationError):
        cli.main(["serve", "--host", "0.0.0.0"])  # noqa: S104


def test_tenant_id_from_env_matches_derivation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from parsimony.tenant import derive_tenant

    monkeypatch.setenv("PARSIMONY_TENANT_CREDENTIAL", "sk-live-key")
    monkeypatch.delenv("PARSIMONY_SALT", raising=False)
    rc = cli.main(["tenant-id"])
    assert rc == 0
    printed = capsys.readouterr().out.strip()
    assert printed == derive_tenant([("x-api-key", "sk-live-key")], salt="")
    assert "sk-live-key" not in printed  # the raw credential is never echoed


def test_tenant_id_missing_credential_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PARSIMONY_TENANT_CREDENTIAL", raising=False)
    monkeypatch.setattr("sys.stdin.readline", lambda: "")  # empty stdin
    rc = cli.main(["tenant-id"])
    assert rc == 1
    assert "no credential" in capsys.readouterr().err
