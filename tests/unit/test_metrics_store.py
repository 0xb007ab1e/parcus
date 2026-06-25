"""Tests for the persistent metrics store and the stats renderer."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from parcus.obs import SavingsEvent, SqliteMetricsSink, StageStat, render_stats


def _event(stages: tuple[StageStat, ...], *, cache: str = "miss", tenant: str = "") -> SavingsEvent:
    return SavingsEvent(
        request_id="r",
        dialect="anthropic",
        cache=cache,
        canonicalized=True,
        tokens_before=10,
        tokens_after=8,
        status_code=200,
        duration_ms=1.0,
        stages=stages,
        tenant=tenant,
    )


class TestSqliteMetricsSink:
    def test_records_and_aggregates_stages(self) -> None:
        sink = SqliteMetricsSink()
        sink.record(_event((StageStat("lossless", 10, 8, True),), cache="miss"))
        sink.record(_event((StageStat("lossless", 10, 9, False),), cache="hit"))
        snap = sink.snapshot()
        assert snap["requests"] == 2
        assert snap["cache_hits"] == 1
        assert snap["tokens_before"] == 20
        assert snap["stages"]["lossless"]["checked"] == 2
        assert snap["stages"]["lossless"]["accuracy"] == 0.5

    def test_records_latest_eval_per_kind(self) -> None:
        sink = SqliteMetricsSink()
        sink.record_eval("retrieval", 0.5, False)
        sink.record_eval("retrieval", 1.0, True)
        evals = sink.snapshot()["evals"]
        assert evals["retrieval"]["score"] == 1.0  # most recent wins
        assert evals["retrieval"]["passed"] is True

    def test_empty_snapshot(self) -> None:
        snap = SqliteMetricsSink().snapshot()
        assert snap["requests"] == 0
        assert snap["overall_ratio"] == 0.0
        assert snap["stages"] == {}
        assert snap["evals"] == {}
        assert snap["by_tenant"] == {}

    def test_by_tenant_attribution(self) -> None:
        sink = SqliteMetricsSink()
        sink.record(_event((), tenant="t1"))
        sink.record(_event((), tenant="t1"))
        sink.record(_event((), tenant="t2"))
        sink.record(_event(()))  # tenant '' excluded from per-tenant rollup
        by_tenant = sink.snapshot()["by_tenant"]
        assert set(by_tenant) == {"t1", "t2"}
        assert by_tenant["t1"]["requests"] == 2
        assert by_tenant["t1"]["tokens_saved"] == 4  # 2 requests * (10-8)

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = str(tmp_path / "m.sqlite")
        sink = SqliteMetricsSink(path)
        sink.record(_event((StageStat("lossless", 10, 8, True),)))
        sink.close()
        reopened = SqliteMetricsSink(path)
        assert reopened.snapshot()["requests"] == 1
        reopened.close()

    def test_file_is_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "m.sqlite"
        sink = SqliteMetricsSink(str(path))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        sink.close()

    def test_fails_open_after_close(self) -> None:
        sink = SqliteMetricsSink()
        sink.close()
        sink.record(_event((StageStat("lossless", 10, 8, True),)))  # no raise
        sink.record_eval("x", 1.0, True)  # no raise

    def test_migrates_legacy_db_without_tenant_column(self, tmp_path: Path) -> None:
        import sqlite3

        path = str(tmp_path / "legacy.sqlite")
        # A pre-tenant events table (no tenant column).
        legacy = sqlite3.connect(path)
        legacy.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, dialect TEXT NOT NULL, "
            "cache TEXT NOT NULL, tokens_before INTEGER NOT NULL, tokens_after INTEGER NOT NULL, "
            "status_code INTEGER NOT NULL, created_at REAL NOT NULL)"
        )
        legacy.commit()
        legacy.close()
        sink = SqliteMetricsSink(path)  # should add the tenant column on open
        sink.record(_event((), tenant="t1"))  # and accept tenant-tagged writes
        snap = sink.snapshot()
        assert snap["requests"] == 1
        assert snap["by_tenant"]["t1"]["requests"] == 1
        sink.close()


class TestRenderStats:
    def test_renders_totals_stages_and_evals(self) -> None:
        sink = SqliteMetricsSink()
        sink.record(_event((StageStat("lossless", 10, 8, True),)))
        sink.record_eval("retrieval", 0.9, True)
        out = render_stats(sink.snapshot())
        assert "requests=1" in out
        assert "lossless" in out
        assert "eval gates" in out
        assert "retrieval" in out

    def test_renders_per_tenant_section_when_present(self) -> None:
        sink = SqliteMetricsSink()
        sink.record(_event((StageStat("lossless", 10, 8, True),), tenant="acme01"))
        out = render_stats(sink.snapshot())
        assert "per-tenant attribution" in out
        assert "acme01" in out

    def test_omits_per_tenant_section_when_single_tenant(self) -> None:
        sink = SqliteMetricsSink()
        sink.record(_event((StageStat("lossless", 10, 8, True),)))  # tenant ''
        assert "per-tenant attribution" not in render_stats(sink.snapshot())
