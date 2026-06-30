"""Parse the provider's reported token ``usage`` from a forwarded (non-streaming) response.

parcus measures the request with a local tokenizer (an estimate); the provider's response body
carries the **billed** counts and, crucially, its **prompt-cache** numbers. Capturing them turns
"savings" from an estimate into ground truth and makes the provider-cache interaction observable
(did compression keep the cache hit, or bust it?). Read-only and **fail-open**: any parse problem
yields ``None`` — the response is never altered or blocked.

Provider shapes (Messages / Chat Completions ``usage`` objects):
* **Anthropic:** ``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``,
  ``cache_creation_input_tokens``.
* **OpenAI:** ``prompt_tokens``, ``completion_tokens``, ``prompt_tokens_details.cached_tokens``
  (no cache-write concept → 0).
"""

from __future__ import annotations

import json

from parcus.model import Dialect, ProviderUsage

__all__ = ["parse_usage"]


def _int(value: object) -> int:
    """Coerce a JSON number to a non-negative int, else 0 (tolerant of nulls/strings/floats)."""
    if isinstance(value, bool):  # bool is an int subclass — exclude it explicitly
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def parse_usage(dialect: Dialect, content: bytes) -> ProviderUsage | None:
    """Return the provider's :class:`ProviderUsage` from a response body, or ``None``.

    Args:
        dialect: The provider dialect the response came from.
        content: The raw response body bytes.

    Returns:
        The parsed usage, or ``None`` if the body isn't JSON, has no ``usage`` object, or the
        dialect is unknown. Never raises.
    """
    try:
        decoded = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    usage = decoded.get("usage")
    if not isinstance(usage, dict):
        return None

    if dialect is Dialect.ANTHROPIC:
        return ProviderUsage(
            input_tokens=_int(usage.get("input_tokens")),
            output_tokens=_int(usage.get("output_tokens")),
            cache_read_tokens=_int(usage.get("cache_read_input_tokens")),
            cache_write_tokens=_int(usage.get("cache_creation_input_tokens")),
        )
    if dialect is Dialect.OPENAI:
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        return ProviderUsage(
            input_tokens=_int(usage.get("prompt_tokens")),
            output_tokens=_int(usage.get("completion_tokens")),
            cache_read_tokens=_int(cached),
            cache_write_tokens=0,  # OpenAI has no separate cache-write count
        )
    return None
