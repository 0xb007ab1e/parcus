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

__all__ = ["detect", "gemini_model_from_path", "parse", "serialize"]

_ANTHROPIC_ROLES = frozenset({Role.USER, Role.ASSISTANT})
_OPENAI_ROLES = frozenset({Role.SYSTEM, Role.USER, Role.ASSISTANT})

# Gemini uses ``user``/``model`` in ``contents`` (not ``assistant``); map both ways.
_GEMINI_TO_ROLE = {"user": Role.USER, "model": Role.ASSISTANT}
_ROLE_TO_GEMINI = {Role.USER: "user", Role.ASSISTANT: "model"}


def detect(path: str) -> Dialect:
    """Return the provider dialect implied by the request ``path``."""
    if path.endswith("/v1/messages"):
        return Dialect.ANTHROPIC
    if path.endswith("/v1/chat/completions"):
        return Dialect.OPENAI
    if path.endswith(":generateContent") or path.endswith(":streamGenerateContent"):
        return Dialect.GEMINI
    return Dialect.UNKNOWN


def gemini_model_from_path(path: str) -> str | None:
    """Extract the model id from a Gemini path, or ``None`` if it can't be determined.

    Gemini puts the model in the URL, not the body — it is the **last path segment before the
    ``:method`` suffix**, which covers both base models
    (``/v1beta/models/gemini-2.5-flash:generateContent``) and tuned models
    (``/v1beta/tunedModels/my-model:generateContent``). The engine folds this into
    :attr:`~parcus.model.CanonicalRequest.model` so two models never collide in the cache key.
    Returns ``None`` for a pathological path (no ``:method`` suffix, or an empty model segment); the
    engine then declines to cache the request rather than risk a cross-model hit (fail open).
    """
    _prefix, _slash, segment = path.rpartition("/")
    model, sep, method = segment.partition(":")
    if not sep or not model or not method:
        return None
    return model


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
    dialect: Dialect,
    body: dict[str, Any],
    *,
    structured: bool = False,
    model: str | None = None,
) -> CanonicalRequest | None:
    """Parse a provider body into a canonical request, or ``None`` if outside the handled subset.

    Args:
        dialect: The detected dialect.
        body: The decoded JSON request body.
        structured: When ``True``, also accept **structured** messages (block-list content, tool
            calls, tool role) by carrying each such message dict verbatim (``Message.raw``) so it
            round-trips byte-for-byte and optimizations leave it untouched (M1d slice 1). When
            ``False`` (default), only the plain-text subset is accepted; anything else → ``None``.
        model: The model id, when the dialect carries it out-of-body. Used for **Gemini**, whose
            model lives in the URL path (see :func:`gemini_model_from_path`); ignored by
            Anthropic/OpenAI, which read it from the body.

    Returns:
        A :class:`CanonicalRequest` for handled requests, else ``None`` (the engine passes those
        through unmodified).
    """
    if dialect is Dialect.ANTHROPIC:
        return _parse_anthropic(body, structured=structured)
    if dialect is Dialect.OPENAI:
        return _parse_openai(body, structured=structured)
    if dialect is Dialect.GEMINI:
        return _parse_gemini(body, structured=structured, model=model)
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


def _gemini_text(content: object) -> str | None:
    """Return the text of a **single-text-part** Gemini ``Content``, else ``None``.

    Accepts exactly ``{"parts": [{"text": <str>}]}`` (optionally with a ``role`` key). Anything
    richer — multiple parts, a non-text part (``functionCall``/``inlineData``/…), or extra keys on
    the part — is not the text subset, so the caller carries it verbatim (structured) or passes the
    whole request through. Keeping it to one part means re-serialising ``{parts:[{text}]}`` is
    semantically identical to the input.
    """
    if not isinstance(content, dict) or set(content.keys()) not in ({"parts"}, {"role", "parts"}):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        return None
    part = parts[0]
    if isinstance(part, dict) and set(part.keys()) == {"text"} and isinstance(part["text"], str):
        return part["text"]
    return None


def _gemini_role(value: object) -> Role | None:
    """Map a Gemini ``contents`` role string (``user``/``model``) to a canonical Role, else None."""
    return _GEMINI_TO_ROLE.get(value) if isinstance(value, str) else None


def _gemini_message(item: dict[str, Any], structured: bool) -> Message | None:
    """Build a canonical message from a Gemini ``contents`` item, or ``None`` if unhandled.

    A plain ``{"role": "user"|"model", "parts": [{"text": str}]}`` decomposes into spans so the
    tiers apply. Any other shape (tool parts, multiple parts, unknown role) is carried verbatim as
    ``raw`` when ``structured`` is set; otherwise it is unhandled → ``None`` (pass the request
    through). ``role`` maps ``model``→assistant; a structured item with an unknown/absent role is
    tagged ``user`` for the cache-key view only — it re-serialises from ``raw`` regardless.
    """
    if set(item.keys()) == {"role", "parts"}:
        role = _gemini_role(item.get("role"))
        text = _gemini_text(item)
        if role is not None and text is not None:
            return Message(role=role, spans=classify_spans(text))
    if structured:
        return Message(role=_gemini_role(item.get("role")) or Role.USER, spans=(), raw=item)
    return None


def _parse_gemini(
    body: dict[str, Any], *, structured: bool, model: str | None
) -> CanonicalRequest | None:
    """Parse a Gemini ``generateContent`` body; ``model`` comes from the URL path (not the body).

    Handles the text subset (``systemInstruction`` + ``contents`` of single-text-part turns) and,
    when ``structured``, carries richer turns verbatim. Returns ``None`` (pass through unmodified)
    for a non-text-only ``systemInstruction``, a missing/!list ``contents``, or any turn it can't
    safely round-trip.
    """
    raw_system = body.get("systemInstruction")
    if raw_system is None:
        system: str | None = None
    else:
        system = _gemini_text(raw_system)
        if system is None:
            return None  # systemInstruction present but not text-only → pass through
    contents = body.get("contents")
    if not isinstance(contents, list):
        return None
    messages: list[Message] = []
    for item in contents:
        if not isinstance(item, dict):
            return None
        message = _gemini_message(item, structured)
        if message is None:
            return None
        messages.append(message)
    return CanonicalRequest(
        dialect=Dialect.GEMINI,
        model=model,
        messages=tuple(messages),
        system=system,
        stream=False,  # Gemini streams by endpoint (:streamGenerateContent), not a body flag
        tools_json=_tools_json(body),
    )


def _serialize_gemini(request: CanonicalRequest, new_body: dict[str, Any]) -> dict[str, Any]:
    """Re-serialise a canonical Gemini request: rebuild ``contents`` + ``systemInstruction``.

    Text turns render as ``{"role": <user|model>, "parts": [{"text": …}]}``; structured turns are
    written back verbatim from ``raw``. Gemini carries no in-request cache breakpoint (its context
    cache is referenced out-of-band), so there is nothing to inject here. Other fields
    (``generationConfig``, ``tools``, ``cachedContent``, …) are preserved from the original body.
    """
    contents: list[dict[str, Any]] = []
    for m in request.messages:
        if m.raw is not None:
            contents.append(m.raw)
            continue
        contents.append({"role": _ROLE_TO_GEMINI.get(m.role, "user"), "parts": [{"text": m.text}]})
    new_body["contents"] = contents
    if request.system is not None:
        new_body["systemInstruction"] = {"parts": [{"text": request.system}]}
    return new_body


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
    if request.dialect is Dialect.GEMINI:
        return _serialize_gemini(request, new_body)
    breakpoint_at = _anthropic_breakpoint_index(request)
    messages: list[dict[str, Any]] = []
    for i, m in enumerate(request.messages):
        if m.raw is not None:
            # Structured message: verbatim, or with a cache_control breakpoint on its last block.
            messages.append(_mark_raw(m.raw) if i == breakpoint_at else m.raw)
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


def _mark_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Add a ``cache_control`` breakpoint to the last content block of a structured message.

    Anthropic caches up to and including the marked block. Returns the message unchanged if its
    content isn't a non-empty block list (no safe place for a block-level marker → no injection).
    """
    content = raw.get("content")
    if not isinstance(content, list) or not content:
        return raw
    last = content[-1]
    if not isinstance(last, dict):
        return raw
    marked = {**last, "cache_control": {"type": "ephemeral"}}
    return {**raw, "content": [*content[:-1], marked]}


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
