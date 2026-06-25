"""Deterministic cache-key derivation for the exact-match response cache.

The key is a SHA-256 over a canonical serialisation of the request's *semantically salient*
fields. The prompt text is **not stored** anywhere — only this one-way hash — which limits
what the cache persists. An optional per-install ``salt`` provides domain separation so hashes
are not comparable across installs. Volatile fields that do not change the logical response
(e.g. the ``stream`` flag) are intentionally excluded.

A version prefix (``v1:``) namespaces the algorithm, so changing the derivation in future
cleanly invalidates old entries rather than risking a wrong hit.
"""

from __future__ import annotations

import hashlib
import json

from parcus.model import CanonicalRequest

__all__ = ["KEY_VERSION", "compute_key"]

KEY_VERSION = "v1"


def compute_key(request: CanonicalRequest, salt: str = "") -> str:
    """Return the stable cache key for ``request``.

    Args:
        request: The canonical request to key on.
        salt: Optional per-install salt for domain separation (default none).

    Returns:
        A string of the form ``"v1:<hex sha-256>"``.
    """
    payload = {
        "v": KEY_VERSION,
        "dialect": request.dialect.value,
        "model": request.model,
        "system": request.system,
        "messages": [[m.role.value, m.text] for m in request.messages],
        "tools": request.tools_json,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256()
    if salt:
        digest.update(salt.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(blob.encode("utf-8"))
    return f"{KEY_VERSION}:{digest.hexdigest()}"
