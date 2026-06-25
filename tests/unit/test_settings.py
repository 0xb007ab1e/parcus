"""Tests for environment-driven settings and the public-bind guard."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from parsimony.config import Settings


def test_secure_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.host == "127.0.0.1"
    assert s.port == 8787
    assert s.cache is True
    assert s.lossless is True
    assert s.filler is False  # lossy passes off by default
    assert s.redact is True
    assert s.tailnet_ip is None


def test_rejects_public_bind() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, host="0.0.0.0")  # noqa: S104


def test_rejects_empty_bind() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, host="")


def test_nocache_patterns_parsing() -> None:
    s = Settings(_env_file=None, cache_nocache_patterns="a, b ,")
    assert s.nocache_patterns() == ["a", "b"]

    assert Settings(_env_file=None).nocache_patterns() == []


def test_allowed_tenants_parsing() -> None:
    s = Settings(_env_file=None, multi_tenant=True, allowed_tenants="abc, def ,")
    assert s.allowed_tenant_set() == frozenset({"abc", "def"})
    assert Settings(_env_file=None).allowed_tenant_set() == frozenset()


def test_allow_list_without_multi_tenant_is_rejected() -> None:
    # Fail fast: an allow-list keys on the tenant id, which only exists in multi-tenant mode.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, allowed_tenants="abc", multi_tenant=False)


def test_allow_list_with_multi_tenant_is_accepted() -> None:
    s = Settings(_env_file=None, allowed_tenants="abc", multi_tenant=True)
    assert s.allowed_tenant_set() == frozenset({"abc"})


def test_rate_limit_disabled_by_default() -> None:
    assert Settings(_env_file=None).rate_limit() is None


def test_rate_limit_built_when_configured() -> None:
    s = Settings(_env_file=None, rate_limit_per_minute=120, rate_limit_burst=30)
    limit = s.rate_limit()
    assert limit is not None
    assert limit.capacity == 30
    assert limit.refill_per_sec == 2.0  # 120/min


def test_negative_rate_limit_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rate_limit_per_minute=-1)
