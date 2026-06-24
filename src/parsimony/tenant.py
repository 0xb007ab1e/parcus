"""Server-side tenant derivation for multi-tenant (hosted) mode.

In hosted mode a single proxy instance serves multiple principals. Tenant identity is derived
**server-side from the inbound credential** the caller already presents (the provider API key /
bearer token) — *never* from a client-supplied tenant field. Trusting a client-asserted tenant
id would be Broken Object Level Authorization (OWASP API1 / BOLA): any caller could read another
tenant's cached data simply by claiming their id.

The derived id is an opaque, salted SHA-256 digest. The raw credential is never stored or
logged — only this short digest, used to namespace per-tenant state (the response cache today;
memory, metrics, and quotas in later slices) so one tenant can never observe another's data.
Single-tenant (local) mode never calls this: the tenant is the empty string and nothing is
namespaced, preserving today's behaviour exactly.
"""

from __future__ import annotations

import hashlib

__all__ = ["ANONYMOUS_TENANT", "derive_tenant", "is_authorized"]

# Used when multi-tenant mode is on but the request carries no recognised credential. Such
# requests share one bucket rather than leaking into a credentialed tenant's namespace.
ANONYMOUS_TENANT = "anon"

# Inbound headers that carry a principal's credential, in priority order. x-api-key is
# Anthropic's; authorization is OpenAI's (and the generic bearer scheme).
_CREDENTIAL_HEADERS = ("x-api-key", "authorization")

# Truncated digest length (hex chars). 16 hex chars = 64 bits of the SHA-256 — ample to avoid
# collisions across realistic tenant counts while keeping the id compact and content-free.
_DIGEST_LEN = 16


def derive_tenant(headers: list[tuple[str, str]], *, salt: str = "") -> str:
    """Derive an opaque tenant id from the inbound credential (server-side; never client input).

    The credential is taken from the request's own auth header — the same secret already sent
    upstream — so the tenant is the authenticated principal, not a value the client can choose.

    Args:
        headers: Inbound request headers (case-insensitive names).
        salt: Per-install salt mixed into the digest for domain separation (same role as the
            cache-key salt). Does not need to be secret; it scopes ids to this install.

    Returns:
        A short hex digest identifying the credential, or :data:`ANONYMOUS_TENANT` when no
        credential header is present. Two requests map to the same tenant id **iff** they carry
        the same credential.
    """
    lower = {k.lower(): v for k, v in headers}
    for name in _CREDENTIAL_HEADERS:
        credential = lower.get(name)
        if credential:
            digest = hashlib.sha256(f"{salt}|{credential}".encode()).hexdigest()
            return digest[:_DIGEST_LEN]
    return ANONYMOUS_TENANT


def is_authorized(tenant: str, allowed: frozenset[str]) -> bool:
    """Return whether a derived tenant may use the proxy (edge authorization, fail-closed).

    Optional defence-in-depth on top of the provider's own authentication of the forwarded
    credential: it lets a hosted operator restrict (and revoke) *which* principals may use this
    instance without rotating provider keys.

    Args:
        tenant: The tenant id from :func:`derive_tenant`.
        allowed: The configured allow-list of permitted tenant ids.

    Returns:
        ``True`` if ``allowed`` is empty (no edge restriction — open; the provider still
        authenticates the forwarded credential), otherwise ``True`` only when ``tenant`` is a
        member. A non-empty allow-list is **fail-closed**: an unlisted or anonymous tenant is
        denied. Membership compares non-secret digests, so it need not be constant-time — an
        attacker still cannot produce a listed id without holding the matching credential.
    """
    if not allowed:
        return True
    return tenant in allowed
