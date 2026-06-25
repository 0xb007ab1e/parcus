#!/usr/bin/env python3
"""Hermetic Markdown link checker — fails on broken *relative* links and #anchors.

Validates internal documentation links only: relative file/dir targets must exist, and any
``#fragment`` must resolve to a heading in the target file (GitHub-style slug). External links
(http/https/mailto/tel) are intentionally **not** fetched — CI stays deterministic and offline
(master §4: no network reliance). Run from the repo root:

    python scripts/check_links.py

Exits 0 when every relative link resolves, 1 (listing each break) otherwise.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "#!")


def _slug(heading: str) -> str:
    """Approximate GitHub's heading-anchor slug: lowercase, drop punctuation, spaces -> '-'."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)  # drop punctuation (keep word chars, space, hyphen)
    return re.sub(r"\s+", "-", text)


def _strip_code_fences(lines: list[str]) -> list[str]:
    """Blank out fenced code blocks so links inside them aren't treated as real links."""
    out: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def _anchors(path: Path) -> set[str]:
    """Return the set of GitHub-style anchor slugs for a markdown file's headings."""
    slugs: dict[str, int] = {}
    anchors: set[str] = set()
    for line in _strip_code_fences(path.read_text(encoding="utf-8").splitlines()):
        m = _HEADING.match(line)
        if not m:
            continue
        base = _slug(m.group(2))
        n = slugs.get(base, 0)
        slugs[base] = n + 1
        anchors.add(base if n == 0 else f"{base}-{n}")  # GitHub disambiguates dupes with -N
    return anchors


def _markdown_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md", "*.markdown"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(p) for p in out.stdout.split() if p]


def check() -> list[str]:
    """Return a list of broken-link descriptions (empty when all relative links resolve)."""
    errors: list[str] = []
    for md in _markdown_files():
        lines = _strip_code_fences(md.read_text(encoding="utf-8").splitlines())
        for lineno, line in enumerate(lines, 1):
            for target in _LINK.findall(line):
                if target.startswith(_SKIP_SCHEMES):
                    continue
                path_part, _, fragment = target.partition("#")
                dest = md.parent / path_part if path_part else md
                if path_part and not dest.exists():
                    errors.append(f"{md}:{lineno}: missing target -> {target}")
                    continue
                if fragment and dest.is_file() and dest.suffix in (".md", ".markdown"):
                    if fragment not in _anchors(dest):
                        errors.append(f"{md}:{lineno}: no anchor '#{fragment}' in {dest}")
    return errors


def main() -> int:
    """Run the check; print results; return a process exit code."""
    errors = check()
    if errors:
        print(f"Broken relative links ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        return 1
    print(f"Markdown link check: OK ({len(_markdown_files())} files, no broken relative links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
