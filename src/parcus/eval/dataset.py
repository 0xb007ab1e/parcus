"""Evaluation samples and a JSONL dataset loader.

A dataset file is JSON-lines; each line is an object with a ``body`` (a provider request body)
and either an explicit ``dialect`` or a ``path`` to infer it from, plus an optional ``name``::

    {"name": "verbose-system", "path": "/v1/messages", "body": {"model": "...", ...}}
    {"name": "openai-chat", "dialect": "openai", "body": {"messages": [...]}}

No real secrets or personal data should be placed in committed datasets (master §5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parcus.model import Dialect
from parcus.proxy.dialects import detect

__all__ = ["Sample", "load_jsonl"]


@dataclass(frozen=True, slots=True)
class Sample:
    """One evaluation request.

    Args:
        name: Human-readable identifier.
        dialect: The provider dialect of ``body``.
        body: The decoded provider request body.
    """

    name: str
    dialect: Dialect
    body: dict[str, Any]


def _dialect_of(obj: dict[str, Any]) -> Dialect:
    if "dialect" in obj:
        return Dialect(obj["dialect"])
    if "path" in obj:
        return detect(str(obj["path"]))
    return Dialect.UNKNOWN


def load_jsonl(path: str | Path) -> tuple[Sample, ...]:
    """Load samples from a JSON-lines dataset file.

    Args:
        path: Path to the ``.jsonl`` dataset.

    Returns:
        The parsed samples in file order (blank lines are ignored).
    """
    samples: list[Sample] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        samples.append(
            Sample(
                name=str(obj.get("name", f"sample-{line_number}")),
                dialect=_dialect_of(obj),
                body=obj["body"],
            )
        )
    return tuple(samples)
