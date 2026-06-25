"""Unit tests for server-side tenant derivation and edge authorization."""

from __future__ import annotations

from parcus.tenant import ANONYMOUS_TENANT, derive_tenant, is_authorized


def test_no_credential_is_anonymous() -> None:
    assert derive_tenant([("content-type", "application/json")]) == ANONYMOUS_TENANT


def test_same_credential_same_tenant() -> None:
    a = derive_tenant([("x-api-key", "sk-abc")])
    b = derive_tenant([("X-API-Key", "sk-abc")])  # header name is case-insensitive
    assert a == b
    assert a != ANONYMOUS_TENANT


def test_different_credentials_different_tenants() -> None:
    a = derive_tenant([("x-api-key", "sk-abc")])
    b = derive_tenant([("x-api-key", "sk-xyz")])
    assert a != b


def test_authorization_header_is_recognised() -> None:
    a = derive_tenant([("authorization", "Bearer tok-1")])
    assert a != ANONYMOUS_TENANT
    assert a != derive_tenant([("authorization", "Bearer tok-2")])


def test_x_api_key_takes_priority_over_authorization() -> None:
    # When both are present, the Anthropic key wins (deterministic precedence).
    both = derive_tenant([("authorization", "Bearer tok"), ("x-api-key", "sk-abc")])
    assert both == derive_tenant([("x-api-key", "sk-abc")])


def test_salt_changes_the_digest() -> None:
    creds = [("x-api-key", "sk-abc")]
    assert derive_tenant(creds, salt="one") != derive_tenant(creds, salt="two")


def test_digest_is_opaque_and_short() -> None:
    digest = derive_tenant([("x-api-key", "sk-secret-value")])
    assert "sk-secret-value" not in digest  # the raw credential never appears
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


class TestIsAuthorized:
    def test_empty_allow_list_is_open(self) -> None:
        assert is_authorized("anytenant", frozenset()) is True
        assert is_authorized(ANONYMOUS_TENANT, frozenset()) is True

    def test_listed_tenant_passes(self) -> None:
        assert is_authorized("abc123", frozenset({"abc123", "def456"})) is True

    def test_unlisted_tenant_denied(self) -> None:
        assert is_authorized("nope", frozenset({"abc123"})) is False

    def test_anonymous_denied_when_allow_list_set(self) -> None:
        # A non-empty allow-list is fail-closed: a credential-less request never passes.
        assert is_authorized(ANONYMOUS_TENANT, frozenset({"abc123"})) is False
