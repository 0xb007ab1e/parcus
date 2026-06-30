"""Tests for the per-tenant key-epoch stores (monotonic; SQLite variant persists)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from parcus.cache.epoch import EpochStore, InMemoryEpochStore, SqliteEpochStore


def _stores(tmp_path: Path) -> list[EpochStore]:
    return [InMemoryEpochStore(), SqliteEpochStore(str(tmp_path / "epoch.db"))]


class TestEpochStores:
    def test_default_epoch_is_zero(self, tmp_path: Path) -> None:
        for store in _stores(tmp_path):
            assert store.epoch("t") == 0

    def test_bump_increments_and_returns(self, tmp_path: Path) -> None:
        for store in _stores(tmp_path):
            assert store.bump("t") == 1
            assert store.bump("t") == 2
            assert store.epoch("t") == 2

    def test_monotonic_only_increases(self, tmp_path: Path) -> None:
        for store in _stores(tmp_path):
            seen = [store.bump("t") for _ in range(5)]
            assert seen == [1, 2, 3, 4, 5]  # strictly increasing, never resets or decreases

    def test_tenants_are_independent(self, tmp_path: Path) -> None:
        for store in _stores(tmp_path):
            store.bump("a")
            store.bump("a")
            store.bump("b")
            assert store.epoch("a") == 2
            assert store.epoch("b") == 1
            assert store.epoch("c") == 0


class TestSqliteEpochStorePersistence:
    def test_epoch_survives_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "epoch.db")
        store = SqliteEpochStore(path)
        store.bump("t")
        store.bump("t")
        store.close()
        # A fresh store on the same file (a "restart") sees the bumped epoch — no un-shred.
        reopened = SqliteEpochStore(path)
        assert reopened.epoch("t") == 2
        assert reopened.bump("t") == 3  # continues from the persisted value
        reopened.close()

    def test_db_file_is_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "epoch.db"
        SqliteEpochStore(str(path)).close()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("store", [InMemoryEpochStore(), SqliteEpochStore()])
def test_satisfies_protocol(store: EpochStore) -> None:
    assert isinstance(store, EpochStore)
