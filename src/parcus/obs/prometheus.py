"""Render a metrics snapshot in Prometheus text exposition format (opt-in exporter).

Additive to the self-contained JSON stats: lets an existing Prometheus/Grafana stack scrape
per-stage reduction + accuracy and the offline eval scores. Exposed at
``GET /__parcus__/metrics`` when a stats source is wired.
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_prometheus"]


def _family(
    name: str, metric_type: str, help_text: str, samples: list[tuple[dict[str, str], Any]]
) -> list[str]:
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"]
    for labels, value in samples:
        label_str = ""
        if labels:
            label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
        lines.append(f"{name}{label_str} {value}")
    return lines


def render_prometheus(snapshot: dict[str, Any]) -> str:
    """Render an aggregate metrics snapshot as Prometheus exposition text.

    Args:
        snapshot: The aggregate dict (totals + per-stage reduction/accuracy + eval scores).

    Returns:
        Prometheus text-format metrics (one trailing newline).
    """
    out: list[str] = []
    out += _family(
        "parcus_requests_total",
        "counter",
        "Total proxied requests.",
        [({}, snapshot["requests"])],
    )
    out += _family(
        "parcus_cache_hit_rate",
        "gauge",
        "Cache hit rate in [0,1].",
        [({}, snapshot["cache_hit_rate"])],
    )
    out += _family(
        "parcus_tokens_saved_total",
        "counter",
        "Total input tokens saved.",
        [({}, snapshot["tokens_saved"])],
    )
    out += _family(
        "parcus_overall_reduction_ratio",
        "gauge",
        "Overall input-token reduction in [0,1].",
        [({}, snapshot["overall_ratio"])],
    )

    stages = sorted(snapshot.get("stages", {}).items())
    if stages:
        out += _family(
            "parcus_stage_reduction_ratio",
            "gauge",
            "Per-stage token reduction in [0,1].",
            [({"stage": name}, s["reduction"]) for name, s in stages],
        )
        out += _family(
            "parcus_stage_tokens_saved_total",
            "counter",
            "Per-stage tokens saved.",
            [({"stage": name}, s["tokens_saved"]) for name, s in stages],
        )
        accuracy = [
            ({"stage": name}, s["accuracy"]) for name, s in stages if s["accuracy"] is not None
        ]
        if accuracy:
            out += _family(
                "parcus_stage_accuracy_ratio",
                "gauge",
                "Per-stage invariant pass-rate in [0,1].",
                accuracy,
            )

    evals = sorted(snapshot.get("evals", {}).items())
    if evals:
        out += _family(
            "parcus_eval_score",
            "gauge",
            "Latest offline eval score per kind in [0,1].",
            [({"kind": kind}, e["score"]) for kind, e in evals],
        )
        out += _family(
            "parcus_eval_passed",
            "gauge",
            "Latest offline eval gate result (1=pass).",
            [({"kind": kind}, 1 if e["passed"] else 0) for kind, e in evals],
        )
    return "\n".join(out) + "\n"
