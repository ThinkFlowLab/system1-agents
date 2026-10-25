# coding: utf-8
"""Latency summary and the hook for a later Public231-C1-style ceiling.

Nothing here contacts a worker. A workflow run records per-request wall time and leaves the
ceiling unattached. ``gate`` is what a later operator command calls once measured samples exist.
"""

from __future__ import annotations

import math
from typing import Any

CLOCK = "client_request_wall_ms"
PROCEDURE = (
    "Public231 C1 later gate: drop a warmup of 5, keep concurrency at 1, repeat three measured rounds, "
    "then compare p50 and p95 of successful request latencies. "
    "This client's clock is client_request_wall_ms, from the start of the POST through the response body. "
    "System1-Omni's bench clock is post_to_body_json_and_schema_validation; do not treat the two as the same number. "
    "No ceiling is pinned for the content fixture. A p50 from another workload is not this fixture's budget."
)


def percentile(samples: list[float], percent: float) -> float:
    """Nearest-rank percentile. ``percent`` is in 0–100. Rank is ``ceil(percent/100 * n)``, clamped to the sample."""
    if not samples:
        raise ValueError("percentile needs at least one sample")
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_ms(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0, "min_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None, "clock": CLOCK}
    return {
        "n": len(samples),
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "max_ms": max(samples),
        "clock": CLOCK,
    }


def gate(
    samples: list[float],
    *,
    p50_max_ms: float | None = None,
    p95_max_ms: float | None = None,
    min_samples: int = 1,
) -> dict[str, Any]:
    """Enforce optional ceilings. With neither ceiling set, the gate is not attached and does not pass or fail.

    When a ceiling is set, fewer than ``min_samples`` fails closed. Zero-millisecond rule-baseline rows must
    not be passed in: they would clear a ceiling without a worker.
    """
    summary = summarize_ms(samples)
    enforced = p50_max_ms is not None or p95_max_ms is not None
    if not enforced:
        return {"enforced": False, "passed": None, "summary": summary, "failures": [], "procedure": PROCEDURE}
    failures: list[str] = []
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1 when a ceiling is set")
    if len(samples) < min_samples:
        failures.append(f"need at least {min_samples} omni request samples, got {len(samples)}")
    else:
        if p50_max_ms is not None and summary["p50_ms"] > p50_max_ms:
            failures.append(f"p50 {summary['p50_ms']} ms is above {p50_max_ms} ms")
        if p95_max_ms is not None and summary["p95_ms"] > p95_max_ms:
            failures.append(f"p95 {summary['p95_ms']} ms is above {p95_max_ms} ms")
    return {
        "enforced": True,
        "passed": not failures,
        "summary": summary,
        "failures": failures,
        "procedure": PROCEDURE,
    }


def omni_call_latencies(report: dict[str, Any]) -> list[float]:
    """Successful Omni request latencies from a workflow report. Rule rows are left out."""
    samples: list[float] = []
    for stage in ("screen", "fact_check"):
        block = report.get(stage) or {}
        for record in block.get("records") or []:
            if record.get("backend") != "omni":
                continue
            for call in record.get("calls") or []:
                samples.append(float(call["latency_ms"]))
    return samples
