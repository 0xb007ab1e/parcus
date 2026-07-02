"""Detect the provider dialect and parse/serialise between it and the canonical model.

The **text-only subset** (always accepted) decomposes plain ``{role, content: str}`` messages
into spans so the compression tiers apply:

* Anthropic Messages — a string ``system`` (or none) and messages whose ``content`` is a string.
* OpenAI Chat Completions — messages that are exactly ``{role, content:str}`` with role in
  system/user/assistant.

With ``structured=True`` (M1d slice 1), messages outside that subset — content blocks, images,
tool_use/tool_result, OpenAI tool calls / ``tool`` role — are **carried verbatim** (``Message.raw``)
so the round-trip reproduces them byte-for-byte and optimizations leave them untouched. Anything
still unhandled (unknown role, Anthropic block-list ``system``) returns ``None`` from :func:`parse`
— the engine forwards the request **unmodified** (fail open). See
``docs/design/structured-content-parser.md``.
"""

from __future__ import annotations

import json
from typing import Any

from parcus.model import CanonicalRequest, Dialect, Message, Role
from parcus.spans import classify_spans

__all__ = ["detect", "parse", "serialize"]

_ANTHROPIC_ROLES = frozenset({Role.USER, Role.ASSISTANT})
_OPENAI_ROLES = frozenset({Role.SYSTEM, Role.USER, Role.ASSISTANT})


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


def parse(
    dialect: Dialect, body: dict[str, Any], *, structured: bool = False
) -> CanonicalRequest | None:
    """Parse a provider body into a canonical request, or ``None`` if outside the handled subset.

    Args:
        dialect: The detected dialect.
        body: The decoded JSON request body.
        structured: When ``True``, also accept **structured** messages (block-list content, tool
            calls, tool role) by carrying each such message dict verbatim (``Message.raw``) so it
            round-trips byte-for-byte and optimizations leave it untouched (M1d slice 1). When
            ``False`` (default), only the plain-text subset is accepted; anything else → ``None``.

    Returns:
        A :class:`CanonicalRequest` for handled requests, else ``None`` (the engine passes those
        through unmodified).
    """
    if dialect is Dialect.ANTHROPIC:
        return _parse_anthropic(body, structured=structured)
    if dialect is Dialect.OPENAI:
        return _parse_openai(body, structured=structured)
    return None


def _message(
    role: Role, item: dict[str, Any], text_roles: frozenset[Role], structured: bool
) -> Message | None:
    """Build a canonical message, or ``None`` if the item can't be safely round-tripped.

    Plain ``{role, content: str}`` (in ``text_roles``) decomposes into spans so the existing tiers
    apply. Any other shape is carried verbatim as ``raw`` when ``structured`` is set (round-trips
    exactly, untouched by optimization); otherwise it is unhandled → ``None``.
    """
    content = item.get("content")
    if role in text_roles and set(item.keys()) == {"role", "content"} and isinstance(content, str):
        return Message(role=role, spans=classify_spans(content))
    if structured:
        return Message(role=role, spans=(), raw=item)
    return None


def _parse_messages(
    body: dict[str, Any], text_roles: frozenset[Role], structured: bool
) -> list[Message] | None:
    raw = body.get("messages")
    if not isinstance(raw, list):
        return None
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = _role(item.get("role"))
        if role is None:  # unknown role → can't safely round-trip; pass the whole request through
            return None
        message = _message(role, item, text_roles, structured)
        if message is None:
            return None
        messages.append(message)
    return messages


def _parse_anthropic(body: dict[str, Any], *, structured: bool) -> CanonicalRequest | None:
    system = body.get("system")
    if system is not None and not isinstance(system, str):
        return None  # system-as-block-list is deferred (see the design note); pass through
    messages = _parse_messages(body, _ANTHROPIC_ROLES, structured)
    if messages is None:
        return None
    return CanonicalRequest(
        dialect=Dialect.ANTHROPIC,
        model=body.get("model"),
        messages=tuple(messages),
        system=system,
        stream=bool(body.get("stream", False)),
        tools_json=_tools_json(body),
    )


def _parse_openai(body: dict[str, Any], *, structured: bool) -> CanonicalRequest | None:
    messages = _parse_messages(body, _OPENAI_ROLES, structured)
    if messages is None:
        return None
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
        if m.raw is not None:
            messages.append(m.raw)  # structured message: reproduce the original dict verbatim
            continue
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
