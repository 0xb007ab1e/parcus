"""Tests for the CLI / composition root (without launching a server)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from parsimony import cli
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


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_serve_rejects_public_bind() -> None:
    with pytest.raises(ValidationError):
        cli.main(["serve", "--host", "0.0.0.0"])  # noqa: S104
