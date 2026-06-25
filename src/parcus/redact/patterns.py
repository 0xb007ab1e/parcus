"""Bounded, model-free patterns for detecting secrets and PII.

Patterns are split into two classes:

* **secret** (credentials/keys) — matching these both masks the span *and* triggers the
  cache no-cache bypass (we refuse to persist credential-bearing requests).
* **pii** (e.g. e-mail addresses) — masked in logs/derived content, but does **not** bypass
  caching (otherwise ordinary requests that merely mention an address would never cache).

All expressions are linear (no nested/ambiguous quantifiers) to avoid ReDoS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["DEFAULT_PATTERNS", "SecretPattern"]


@dataclass(frozen=True, slots=True)
class SecretPattern:
    """A named detection rule.

    Args:
        name: Category label used in placeholders and reports (e.g. ``"api_key"``).
        regex: Compiled, linear-time pattern.
        is_secret: True for credentials (mask + cache-bypass); False for PII (mask only).
    """

    name: str
    regex: re.Pattern[str]
    is_secret: bool = True


# Credentials / keys -------------------------------------------------------------------
# Order: Anthropic before the generic OpenAI-style key; the OpenAI rule also excludes
# the "sk-ant-" prefix so the two never overlap.
DEFAULT_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    SecretPattern("openai_key", re.compile(r"sk-(?!ant-)[A-Za-z0-9]{20,}")),
    SecretPattern("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    SecretPattern("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    SecretPattern("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    SecretPattern(
        "private_key_block",
        re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    ),
    SecretPattern("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}")),
    SecretPattern(
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|access[_-]?key|password|passwd|pwd|token)\b"
            r"\s*[:=]\s*['\"]?[^\s'\"]{8,}"
        ),
    ),
    # PII (masked, but not treated as a credential) ------------------------------------
    SecretPattern(
        "email",
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
        is_secret=False,
    ),
)
