"""A small built-in corpus used for a quick eval run and for tests.

These are synthetic (no secrets/PII) and intentionally contain redundant whitespace so the
lossless pass has something to remove, plus a fenced-code case (must be preserved) and a
content-block case (must pass through untouched).
"""

from __future__ import annotations

from parsimony.eval.dataset import Sample
from parsimony.model import Dialect

__all__ = ["BUILTIN_SAMPLES"]

BUILTIN_SAMPLES: tuple[Sample, ...] = (
    Sample(
        name="verbose-system",
        dialect=Dialect.ANTHROPIC,
        body={
            "model": "claude-sonnet-4-6",
            "system": "You are a helpful assistant.   \n\n\n\n\nBe concise.   ",
            "messages": [{"role": "user", "content": "Summarize the plan.   \n\n\n\n"}],
        },
    ),
    Sample(
        name="multi-turn",
        dialect=Dialect.ANTHROPIC,
        body={
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Question one.   \n\n\n\n"},
                {"role": "assistant", "content": "Answer one.   "},
                {"role": "user", "content": "Question two.   \n\n\n\n\n"},
            ],
        },
    ),
    Sample(
        name="with-code",
        dialect=Dialect.ANTHROPIC,
        body={
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain this:   \n\n\n\n```python\nx = 1   \ny = 2   \n```"
                        "\n\n\n\nThanks.   "
                    ),
                }
            ],
        },
    ),
    Sample(
        name="openai-chat",
        dialect=Dialect.OPENAI,
        body={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be terse.   \n\n\n\n"},
                {"role": "user", "content": "Hello there.   "},
            ],
        },
    ),
    Sample(
        name="passthrough-blocks",
        dialect=Dialect.ANTHROPIC,
        body={
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "structured content"}]}
            ],
        },
    ),
)
