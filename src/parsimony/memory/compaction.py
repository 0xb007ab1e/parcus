"""Context compaction (Tracks B & C): drop older turns, inject a compact replacement.

Replaces a long history with a recent window plus either the relevant earlier context retrieved
from memory (Track B, query-dependent) or a rolling summary of the dropped turns (Track C,
query-independent) — reducing input tokens. Safety rules so the rewritten request stays valid:

* keep at least ``keep_recent`` messages, extended back so the window **starts on a user
  message** (Anthropic requires that; a suffix of a valid alternation stays valid);
* inject retrieved context by **prepending it to that first kept user message** (an immutable
  span), not by adding a message — so role alternation and message structure are preserved;
* if there is nothing older to drop, or retrieval returns nothing, or anything is off, return
  the request **unchanged** (the engine calls this fail-open and never drops context blindly).
"""

from __future__ import annotations

from parsimony.memory.summary import Summarizer
from parsimony.model import CanonicalRequest, Message, Role, Span
from parsimony.ports import MemoryPort

__all__ = ["compact_by_summary", "compact_with_memory"]

_CONTEXT_HEADER = "Relevant earlier context (retrieved from memory):\n"
_SUMMARY_HEADER = "Conversation summary so far:\n"


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
    return _rewrite(request, start, header)


def compact_by_summary(
    request: CanonicalRequest,
    summarizer: Summarizer,
    *,
    keep_recent: int = 4,
    max_items: int = 5,
    min_messages: int = 8,
) -> CanonicalRequest:
    """Replace older turns with one query-independent summary block (Track C).

    Same structural-safety rules as :func:`compact_with_memory`: keep a recent window starting
    on a user message, prepend the summary to that message, and return the request unchanged
    when there is nothing to drop or the summary is empty.

    Args:
        request: The canonical request to compact.
        summarizer: Produces the summary of the dropped older turns.
        keep_recent: Minimum number of most-recent messages to keep verbatim.
        max_items: Max summary points.
        min_messages: Only compact when the request has more than this many messages.

    Returns:
        A new compacted :class:`CanonicalRequest`, or the original if nothing is dropped.
    """
    messages = request.messages
    if len(messages) <= min_messages:
        return request
    start = _window_start(messages, keep_recent)
    if start is None or start == 0:
        return request
    summary = summarizer.summarize(tuple(m.text for m in messages[:start]), max_items=max_items)
    if not summary:
        return request
    return _rewrite(request, start, _SUMMARY_HEADER + summary + "\n\n")


def _rewrite(request: CanonicalRequest, start: int, header: str) -> CanonicalRequest:
    """Drop messages before ``start`` and prepend ``header`` to the first kept message."""
    kept = list(request.messages[start:])
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
