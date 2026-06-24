"""Track B context compaction: drop older turns, inject retrieved context (structurally safe).

Replaces a long history with a recent window plus the relevant earlier context retrieved from
memory — reducing input tokens. Safety rules so the rewritten request stays valid:

* keep at least ``keep_recent`` messages, extended back so the window **starts on a user
  message** (Anthropic requires that; a suffix of a valid alternation stays valid);
* inject retrieved context by **prepending it to that first kept user message** (an immutable
  span), not by adding a message — so role alternation and message structure are preserved;
* if there is nothing older to drop, or retrieval returns nothing, or anything is off, return
  the request **unchanged** (the engine calls this fail-open and never drops context blindly).
"""

from __future__ import annotations

from parsimony.model import CanonicalRequest, Message, Role, Span
from parsimony.ports import MemoryPort

__all__ = ["compact_with_memory"]

_CONTEXT_HEADER = "Relevant earlier context (retrieved from memory):\n"


def _latest_user_text(messages: tuple[Message, ...]) -> str:
    for message in reversed(messages):
        if message.role is Role.USER:
            return message.text
    return messages[-1].text  # pragma: no cover - only called once a user message is known present


def _window_start(messages: tuple[Message, ...], keep_recent: int) -> int | None:
    """Index where the keep-window starts: a user message at/before ``len - keep_recent``."""
    target = max(0, len(messages) - keep_recent)
    for index in range(target, -1, -1):
        if messages[index].role is Role.USER:
            return index
    return None


def compact_with_memory(
    request: CanonicalRequest,
    memory: MemoryPort,
    *,
    keep_recent: int = 4,
    retrieve: int = 3,
    min_messages: int = 8,
) -> CanonicalRequest:
    """Return a compacted request, or ``request`` unchanged when compaction is not safe/worthwhile.

    Args:
        request: The canonical request to compact.
        memory: The memory to retrieve relevant earlier context from.
        keep_recent: Minimum number of most-recent messages to keep verbatim.
        retrieve: Max retrieved snippets to inject.
        min_messages: Only compact when the request has more than this many messages.

    Returns:
        A new compacted :class:`CanonicalRequest`, or the original if nothing is dropped.
    """
    messages = request.messages
    if len(messages) <= min_messages:
        return request
    start = _window_start(messages, keep_recent)
    if start is None or start == 0:
        return request  # no valid window, or nothing older to drop
    retrieved = memory.relevant(_latest_user_text(messages), limit=retrieve)
    if not retrieved:
        return request  # never drop context blindly

    header = _CONTEXT_HEADER + "\n".join(f"- {snippet}" for snippet in retrieved) + "\n\n"
    kept = list(messages[start:])
    first = kept[0]
    kept[0] = Message(role=first.role, spans=(Span(header, mutable=False), *first.spans))
    return CanonicalRequest(
        dialect=request.dialect,
        model=request.model,
        messages=tuple(kept),
        system=request.system,
        stream=request.stream,
        tools_json=request.tools_json,
    )
