"""Render an aggregate metrics snapshot as a human-readable report."""

from __future__ import annotations

from typing import Any

__all__ = ["render_stats"]


def render_stats(snapshot: dict[str, Any]) -> str:
    """Render a metrics snapshot (from a sink's ``snapshot()``) as a text table.

    Args:
        snapshot: The aggregate dict (totals + per-stage reduction/accuracy + eval scores).

    Returns:
        A multi-line, human-readable report.
    """
    lines = [
        f"requests={snapshot['requests']}  "
        f"cache_hit_rate={snapshot['cache_hit_rate'] * 100:.1f}%  "
        f"tokens {snapshot['tokens_before']}->{snapshot['tokens_after']} "
        f"(saved {snapshot['tokens_saved']}, {snapshot['overall_ratio'] * 100:.1f}%)",
        "",
        f"{'stage':<14} {'before':>8} {'after':>8} {'reduction':>10} {'accuracy':>9}",
        "-" * 54,
    ]
    for name, stage in sorted(snapshot["stages"].items()):
        accuracy = "n/a" if stage["accuracy"] is None else f"{stage['accuracy'] * 100:.1f}%"
        lines.append(
            f"{name:<14} {stage['tokens_before']:>8} {stage['tokens_after']:>8} "
            f"{stage['reduction'] * 100:>9.1f}% {accuracy:>9}"
        )
    if snapshot.get("evals"):
        lines.append("")
        lines.append("eval gates (offline accuracy):")
        for kind, result in sorted(snapshot["evals"].items()):
            verdict = "PASS" if result["passed"] else "FAIL"
            lines.append(f"  {kind:<20} score={result['score'] * 100:.1f}%  {verdict}")
    return "\n".join(lines)
