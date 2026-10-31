"""Choose empirical browser-deferral cutoffs from saved ``decision_ticks.json`` files.

This is deliberately an offline helper: it only reads JSON artifacts and reports the threshold that defers the
requested lowest-confidence fraction of the observed steps.  It does not claim that a decision confidence is a
calibrated probability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_confidences(paths: list[Path]) -> list[float]:
    """Read finite confidence values from complete artifacts or bare tick lists."""
    values: list[float] = []
    for path in paths:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        ticks = payload.get("ticks", []) if isinstance(payload, dict) else payload
        if not isinstance(ticks, list):
            raise ValueError(f"{path}: expected a ticks list")
        for tick in ticks:
            confidence = tick.get("confidence") if isinstance(tick, dict) else None
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence):
                values.append(float(confidence))
    if not values:
        raise ValueError("no finite confidence values found")
    return values


def calibrate(confidences: list[float], rates: list[float]) -> list[dict[str, float | int]]:
    """Return the strict-below threshold and actual count for each requested fraction.

    Ties are never split: the reported actual rate may be lower than requested, which makes a command's routing
    behaviour reproducible instead of depending on artifact order.
    """
    ordered = sorted(confidences)
    if not ordered:
        raise ValueError("at least one confidence is required")
    rows: list[dict[str, float | int]] = []
    for rate in rates:
        if not 0 <= rate <= 1 or not math.isfinite(rate):
            raise ValueError("rates must be finite values from 0 through 1")
        requested = math.floor(len(ordered) * rate)
        if requested == 0:
            threshold = ordered[0]
        else:
            threshold = ordered[min(requested, len(ordered) - 1)]
        count = sum(value < threshold for value in ordered)
        rows.append(
            {
                "requested_rate": rate,
                "defer_below": threshold,
                "deferred_steps": count,
                "total_steps": len(ordered),
                "actual_rate": count / len(ordered),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate offline confidence-gated browser deferral.")
    parser.add_argument("artifacts", type=Path, nargs="+", help="saved decision_ticks.json file(s)")
    parser.add_argument(
        "--rate",
        action="append",
        type=float,
        default=None,
        help="requested deferral fraction, repeatable (default: 0.1, 0.15, 0.25)",
    )
    args = parser.parse_args()
    rates = args.rate or [0.1, 0.15, 0.25]
    print(json.dumps({"calibration": calibrate(read_confidences(args.artifacts), rates)}, indent=2))


if __name__ == "__main__":
    main()
