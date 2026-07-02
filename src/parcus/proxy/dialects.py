"""Detect the provider dialect and parse/serialise between it and the canonical model.

M1 supports a deliberately **conservative, text-only subset** so the round-trip can never
corrupt a request:

* Anthropic Messages — a string ``system`` (or none) and messages whose ``content`` is a
  string.
* OpenAI Chat Completions — messages that are exactly ``{role, content:str}`` with role in
  system/user/assistant.

Anything outside the subset (content blocks, images, tool_use/tool_result, ``tool`` role,
extra per-message keys) returns ``None`` from :func:`parse` — the engine then forwards the
request **unmodified** (fail open). Broadening this to structured content is the top M2 task.
"""

from __future__ import annotations

import json
from typing import Any

from parcus.model import CanonicalRequest, Dialect, Message, Role
from parcus.spans import classify_spans

__all__ = ["detect", "parse", "serialize"]

_ANTHROPIC_ROLES = {Role.USER, Role.ASSISTANT}
_OPENAI_ROLES = {Role.SYSTEM, Role.USER, Role.ASSISTANT}


def detect(path: str) -> Dialect:
    """Return the provider dialect implied by the request ``path``."""
    if path.endswith("/v1/messages"):
        return Dialect.ANTHROPIC
    if path.endswith("/v1/chat/completions"):
        return Dialect.OPENAI
    return Dialect.UNKNOWN


def _role(value: object) -> Role | None:
    try:
        return Role(value)  # type: ignore[arg-type]
    except ValueError:
        return None


def _tools_json(body: dict[str, Any]) -> str | None:
    tools = body.get("tools")
    if tools is None:
        return None
    return json.dumps(tools, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def parse(dialect: Dialect, body: dict[str, Any]) -> CanonicalRequest | None:
    """Parse a provider body into a canonical request, or ``None`` if outside the M1 subset.

    Args:
        dialect: The detected dialect.
        body: The decoded JSON request body.

    Returns:
        A :class:`CanonicalRequest` for safely-handled text-only requests, else ``None``.
    """
    if dialect is Dialect.ANTHROPIC:
        return _parse_anthropic(body)
    if dialect is Dialect.OPENAI:
        return _parse_openai(body)
    return None


def _parse_anthropic(body: dict[str, Any]) -> CanonicalRequest | None:
    system = body.get("system")
    if system is not None and not isinstance(system, str):
        return None
    raw = body.get("messages")
    if not isinstance(raw, list):
        return None
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = _role(item.get("role"))
        content = item.get("content")
        if role not in _ANTHROPIC_ROLES or not isinstance(content, str):
            return None
        messages.append(Message(role=role, spans=classify_spans(content)))
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model=body.get("model"),
        messages=tuple(messages),
        system=system,
        stream=bool(body.get("stream", False)),
        tools_json=_tools_json(body),
    )


def _parse_openai(body: dict[str, Any]) -> CanonicalRequest | None:
    raw = body.get("messages")
    if not isinstance(raw, list):
        return None
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        # Only the plain {role, content} shape is safe to round-trip; anything with
        # tool_calls / tool_call_id / name etc. must pass through untouched.
        if set(item.keys()) - {"role", "content"}:
            return None
        role = _role(item.get("role"))
        content = item.get("content")
        if role not in _OPENAI_ROLES or not isinstance(content, str):
            return None
        messages.append(Message(role=role, spans=classify_spans(content)))
    return CanonicalRequest(
        dialect=Dialect.OPENAI,
        model=body.get("model"),
        messages=tuple(messages),
        system=None,
        stream=bool(body.get("stream", False)),
        tools_json=_tools_json(body),
    )


def serialize(request: CanonicalRequest, original_body: dict[str, Any]) -> dict[str, Any]:
    """Re-serialise a (compressed) canonical request back into its provider body.

    Only the text fields produced from the canonical model are rewritten; all other original
    fields (model, params, tools, stream, …) are preserved. When ``request.cache_breakpoint`` is
    set on an Anthropic request, that message's content is rendered as a single text block
    carrying ``cache_control: {"type": "ephemeral"}`` (the M1b injection); every other message
    keeps the plain-string form, so an un-annotated request serialises byte-identically to before.

    Args:
        request: The canonical request (typically post-compression).
        original_body: The original decoded body, used as the base to preserve extra fields.

    Returns:
        A new body dict ready to serialise and forward upstream.
    """
    new_body = dict(original_body)
    breakpoint_at = _anthropic_breakpoint_index(request)
    messages: list[dict[str, Any]] = []
    for i, m in enumerate(request.messages):
        if i == breakpoint_at:
            content: Any = [
                {"type": "text", "text": m.text, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            content = m.text
        messages.append({"role": m.role.value, "content": content})
    new_body["messages"] = messages
    if request.dialect is Dialect.ANTHROPIC and request.system is not None:
        new_body["system"] = request.system
    return new_body


def _anthropic_breakpoint_index(request: CanonicalRequest) -> int | None:
    """Return the in-range message index to mark with ``cache_control``, else ``None``.

    Only Anthropic requests take an explicit breakpoint; an out-of-range or unset marker renders
    nothing (defensive — the request serialises unchanged).
    """
    idx = request.cache_breakpoint
    if request.dialect is not Dialect.ANTHROPIC or idx is None:
        return None
    if 0 <= idx < len(request.messages):
        return idx
    return None
