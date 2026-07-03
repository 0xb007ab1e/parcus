"""Tests for the CLI / composition root (without launching a server)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from parcus import cli
from parcus.compress import ChainCompressor, NullCompressor
from parcus.config import Settings


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


def test_eval_aggressive_filler_runs_and_passes_guardrail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The model-free guardrail holds for the larger set too, so the gate still passes.
    rc = cli.main(["eval", "--filler", "--aggressive"])
    assert rc == 0
    assert "TOTAL" in capsys.readouterr().out


def test_build_engine_uses_aggressive_fillers_when_configured() -> None:
    from parcus.compress import AGGRESSIVE_FILLERS, FillerCompressor

    engine = cli.build_engine(
        Settings(
            _env_file=None,
            cache=False,
            metrics=False,
            lossless=False,
            filler=True,
            filler_aggressive=True,
        )
    )
    assert isinstance(engine._compressor, FillerCompressor)
    assert engine._compressor._fillers == AGGRESSIVE_FILLERS


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


def test_build_engine_wires_learned_tier_when_enabled() -> None:
    # learned=True appends the Tier-2 LearnedCompressor (model loads lazily, so no extra needed
    # to construct it). With lossless also on, that's a chain.
    from parcus.compress import ChainCompressor, LearnedCompressor

    engine = cli.build_engine(Settings(_env_file=None, cache=False, metrics=False, learned=True))
    assert isinstance(engine._compressor, ChainCompressor)
    only_learned = cli.build_engine(
        Settings(_env_file=None, cache=False, metrics=False, lossless=False, learned=True)
    )
    assert isinstance(only_learned._compressor, LearnedCompressor)


def test_build_engine_wires_llmlingua2_backend_when_configured() -> None:
    # learned_llmlingua2=True selects the LLMLingua-2 backend + its default model on the reducer.
    from parcus.compress import LearnedCompressor
    from parcus.compress.learned import DEFAULT_LLMLINGUA2_MODEL

    engine = cli.build_engine(
        Settings(
            _env_file=None,
            cache=False,
            metrics=False,
            lossless=False,
            learned=True,
            learned_llmlingua2=True,
        )
    )
    assert isinstance(engine._compressor, LearnedCompressor)
    assert engine._compressor._reducer.model_name == DEFAULT_LLMLINGUA2_MODEL


def test_build_engine_honours_explicit_learned_model() -> None:
    from parcus.compress import LearnedCompressor

    engine = cli.build_engine(
        Settings(
            _env_file=None,
            cache=False,
            metrics=False,
            lossless=False,
            learned=True,
            learned_model="my/local-model",
        )
    )
    assert isinstance(engine._compressor, LearnedCompressor)
    assert engine._compressor._reducer.model_name == "my/local-model"


def test_build_engine_wires_rate_limiter_when_configured() -> None:
    engine = cli.build_engine(
        Settings(_env_file=None, cache=False, metrics=False, rate_limit_per_minute=60)
    )
    assert engine._rate_limiter is not None


def test_build_engine_has_no_rate_limiter_by_default() -> None:
    engine = cli.build_engine(Settings(_env_file=None, cache=False, metrics=False))
    assert engine._rate_limiter is None


def test_build_engine_wires_similarity_when_enabled() -> None:
    # Use the dep-free lexical embedder (explicitly acknowledged) so the test needs no model;
    # the safe default is 'local', exercised via settings tests rather than a real model here.
    engine = cli.build_engine(
        Settings(
            _env_file=None,
            cache=False,
            metrics=False,
            similarity_cache=True,
            similarity_embedder="hashing",
            similarity_allow_lexical=True,
        )
    )
    assert engine._similarity is not None


def test_build_engine_has_no_similarity_by_default() -> None:
    engine = cli.build_engine(Settings(_env_file=None, cache=False, metrics=False))
    assert engine._similarity is None


def test_build_engine_wraps_cache_in_encryption_when_enabled() -> None:
    import base64

    from parcus.cache.encryption import EncryptedCache

    key = base64.b64encode(b"\x02" * 32).decode()
    engine = cli.build_engine(
        Settings(_env_file=None, metrics=False, cache_encryption=True, cache_encryption_key=key)
    )
    assert isinstance(engine._cache, EncryptedCache)


def test_build_engine_cache_unencrypted_by_default() -> None:
    from parcus.cache import SqliteCache

    engine = cli.build_engine(Settings(_env_file=None, metrics=False))
    assert isinstance(engine._cache, SqliteCache)


def test_build_engine_uses_per_tenant_dek_in_multi_tenant_mode() -> None:
    import base64

    from parcus.cache.encryption import EncryptedCache, TenantCipherProvider

    key = base64.b64encode(b"\x05" * 32).decode()
    engine = cli.build_engine(
        Settings(
            _env_file=None,
            metrics=False,
            cache_encryption=True,
            cache_encryption_key=key,
            multi_tenant=True,
            cache_shredded_tenants="ghost",
        )
    )
    assert isinstance(engine._cache, EncryptedCache)
    assert isinstance(engine._cache._provider, TenantCipherProvider)
    assert engine._cache._provider.for_tenant("ghost") is None  # shredded


def test_eval_similarity_command_runs(capsys: pytest.CaptureFixture[str]) -> None:
    # The default (lexical) embedder fails the adversarial built-in set -> exit 1, by design:
    # the gate is signalling that the dependency-free embedder is unsafe for semantic caching.
    rc = cli.main(["eval", "--similarity"])
    assert rc == 1
    assert "threshold=" in capsys.readouterr().out


def test_stats_command_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PARCUS_METRICS_PATH", str(tmp_path / "m.sqlite"))
    assert cli.main(["stats"]) == 0  # empty store still renders
    assert "requests=" in capsys.readouterr().out


def test_eval_record_then_stats_shows_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PARCUS_METRICS_PATH", str(tmp_path / "m.sqlite"))
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
    from parcus.tenant import derive_tenant

    monkeypatch.setenv("PARCUS_TENANT_CREDENTIAL", "sk-live-key")
    monkeypatch.delenv("PARCUS_SALT", raising=False)
    rc = cli.main(["tenant-id"])
    assert rc == 0
    printed = capsys.readouterr().out.strip()
    assert printed == derive_tenant([("x-api-key", "sk-live-key")], salt="")
    assert "sk-live-key" not in printed  # the raw credential is never echoed


def test_tenant_id_missing_credential_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PARCUS_TENANT_CREDENTIAL", raising=False)
    monkeypatch.setattr("sys.stdin.readline", lambda: "")  # empty stdin
    rc = cli.main(["tenant-id"])
    assert rc == 1
    assert "no credential" in capsys.readouterr().err


def test_eval_judged_aggressive_filler_passes() -> None:
    # Offline aggressive-filler validation: removing the AGGRESSIVE_FILLERS set preserves the
    # built-in corpus's required content (model-free, CI-safe).
    rc = cli.main(["eval", "--judged", "--filler", "--aggressive"])
    assert rc == 0


def test_eval_learned_skips_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A nonexistent model makes the learned reducer probe fail -> CI-safe skip (exit 0).
    monkeypatch.setenv("PARCUS_LEARNED_MODEL", "/nonexistent/model/path")
    rc = cli.main(["eval", "--learned"])
    assert rc == 0
    assert "skipped" in capsys.readouterr().out.lower()


class TestParseSweepRatios:
    """Keep-ratio sweep spec parsing (pure, model-free)."""

    def test_parses_valid_list(self) -> None:
        assert cli._parse_sweep_ratios("0.3,0.5,0.7,0.9") == [0.3, 0.5, 0.7, 0.9]

    def test_tolerates_spaces_and_trailing_comma(self) -> None:
        assert cli._parse_sweep_ratios(" 0.4 , 0.8 ,") == [0.4, 0.8]

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="keep-ratio must be in"):
            cli._parse_sweep_ratios("0.5,1.5")
        with pytest.raises(ValueError, match="keep-ratio must be in"):
            cli._parse_sweep_ratios("0,0.5")  # 0 is not in (0, 1]

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError):
            cli._parse_sweep_ratios("0.5,abc")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="no keep-ratios"):
            cli._parse_sweep_ratios(" , ")


def test_eval_learned_sweep_rejects_invalid_ratios(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A bad --sweep fails fast (exit 2) before any model load — no 'learned' extra needed.
    rc = cli.main(["eval", "--learned", "--sweep", "0.5,2.0"])
    assert rc == 2
    assert "invalid --sweep" in capsys.readouterr().out.lower()


def test_eval_learned_sweep_skips_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Valid --sweep parses, then the model probe fails on a pinned bad model -> CI-safe skip.
    monkeypatch.setenv("PARCUS_LEARNED_MODEL", "/nonexistent/model/path")
    rc = cli.main(["eval", "--learned", "--sweep"])  # bare flag -> default ratios
    assert rc == 0
    assert "skipped" in capsys.readouterr().out.lower()


def test_eval_learned_honours_llmlingua2_env_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # PARCUS_LEARNED_LLMLINGUA2 selects the v2 backend. Pinned to a nonexistent model so the probe
    # fails deterministically (CI-safe skip) regardless of whether the 'learned' extra is present.
    monkeypatch.setenv("PARCUS_LEARNED_LLMLINGUA2", "true")
    monkeypatch.setenv("PARCUS_LEARNED_MODEL", "/nonexistent/model/path")
    rc = cli.main(["eval", "--learned"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped" in out.lower()
    assert "/nonexistent/model/path" in out  # skip message names the resolved (pinned) model
