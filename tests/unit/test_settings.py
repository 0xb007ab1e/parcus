"""Tests for environment-driven settings and the public-bind guard."""

from __future__ import annotations

from pathlib import Path

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
    assert s.filler_aggressive is False  # conservative set by default
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


def test_similarity_cache_disabled_by_default() -> None:
    s = Settings(_env_file=None)
    assert s.similarity_cache is False
    assert s.similarity_threshold == 0.97


def test_similarity_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, similarity_threshold=1.5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, similarity_threshold=-0.1)


def test_similarity_embedder_defaults_to_local() -> None:
    assert Settings(_env_file=None).similarity_embedder == "local"


def test_similarity_lexical_embedder_rejected_without_acknowledgement() -> None:
    # The unsafe lexical embedder must be explicitly accepted (ADR 0004 fail-closed default).
    with pytest.raises(ValidationError):
        Settings(_env_file=None, similarity_cache=True, similarity_embedder="hashing")


def test_similarity_lexical_embedder_allowed_with_acknowledgement() -> None:
    s = Settings(
        _env_file=None,
        similarity_cache=True,
        similarity_embedder="hashing",
        similarity_allow_lexical=True,
    )
    assert s.similarity_embedder == "hashing"


def test_similarity_unknown_embedder_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, similarity_cache=True, similarity_embedder="openai")


def test_similarity_embedder_unchecked_when_cache_off() -> None:
    # The guard only applies when the cache is enabled — off-by-default config never trips it.
    s = Settings(_env_file=None, similarity_cache=False, similarity_embedder="hashing")
    assert s.similarity_cache is False


def _b64key(nbytes: int = 32) -> str:
    import base64

    return base64.b64encode(b"\x07" * nbytes).decode()


def test_encryption_disabled_by_default() -> None:
    s = Settings(_env_file=None)
    assert s.cache_encryption is False
    assert s.cache_encryption_key_bytes() is None


def test_encryption_key_resolves_to_32_bytes() -> None:
    s = Settings(_env_file=None, cache_encryption=True, cache_encryption_key=_b64key())
    assert s.cache_encryption_key_bytes() == b"\x07" * 32


def test_encryption_enabled_without_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cache_encryption=True)


def test_encryption_wrong_key_length_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cache_encryption=True, cache_encryption_key=_b64key(16))


def test_encryption_invalid_base64_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cache_encryption=True, cache_encryption_key="not base64!!")


def test_encryption_key_from_keyfile(tmp_path: Path) -> None:
    keyfile = tmp_path / "cache.key"
    keyfile.write_text(_b64key())
    s = Settings(_env_file=None, cache_encryption=True, cache_encryption_keyfile=str(keyfile))
    assert s.cache_encryption_key_bytes() == b"\x07" * 32


def test_encryption_missing_keyfile_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None, cache_encryption=True, cache_encryption_keyfile="/no/such/key.file"
        )


def test_encryption_key_is_not_exposed_in_repr() -> None:
    s = Settings(_env_file=None, cache_encryption=True, cache_encryption_key=_b64key())
    assert "\x07" * 32 not in repr(s)  # SecretStr masks the value


def test_previous_keys_resolve_for_rotation() -> None:
    import base64

    prev = base64.b64encode(b"\x09" * 32).decode()
    s = Settings(
        _env_file=None,
        cache_encryption=True,
        cache_encryption_key=_b64key(),
        cache_encryption_previous_keys=prev,
    )
    assert s.cache_encryption_previous_key_bytes() == (b"\x09" * 32,)


def test_no_previous_keys_by_default() -> None:
    s = Settings(_env_file=None, cache_encryption=True, cache_encryption_key=_b64key())
    assert s.cache_encryption_previous_key_bytes() == ()


def test_invalid_previous_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            cache_encryption=True,
            cache_encryption_key=_b64key(),
            cache_encryption_previous_keys="not-valid-base64!!",
        )


def test_learned_disabled_by_default() -> None:
    s = Settings(_env_file=None)
    assert s.learned is False
    assert s.learned_ratio == 0.5


def test_learned_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, learned_ratio=0.0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, learned_ratio=1.5)
