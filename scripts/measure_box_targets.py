# coding: utf-8
"""Offline helper: presence / box hit rate on saved captures with known rectangles (#98).

Uncalibrated thresholds (defaults match the desktop agent). Not a general accuracy claim.

Input JSONL, one object per capture::

    {
      "id": "fixture-1",
      "targets": {
        "Save": {
          "present": true,
          "rect": {"top": 0.4, "bottom": 0.7, "left": 0.55, "right": 0.85}
        }
      },
      "answers": {
        "Save": {
          "presence": 0.91,
          "box": {"top": 0.41, "bottom": 0.69, "left": 0.56, "right": 0.84},
          "box_confidence": 0.88
        }
      }
    }

``answers`` are whatever a prior locate step recorded (or a fake). The script does not call TypeSafe.

Usage::

    python scripts/measure_box_targets.py captures.jsonl
    python scripts/measure_box_targets.py captures.jsonl --presence 0.5 --box-confidence 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    top, left = max(a["top"], b["top"]), max(a["left"], b["left"])
    bottom, right = min(a["bottom"], b["bottom"]), min(a["right"], b["right"])
    if bottom <= top or right <= left:
        return 0.0
    inter = (bottom - top) * (right - left)
    area_a = (a["bottom"] - a["top"]) * (a["right"] - a["left"])
    area_b = (b["bottom"] - b["top"]) * (b["right"] - b["left"])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def centre_in_rect(box: dict[str, float], rect: dict[str, float]) -> bool:
    cx = (box["left"] + box["right"]) / 2.0
    cy = (box["top"] + box["bottom"]) / 2.0
    return rect["left"] <= cx <= rect["right"] and rect["top"] <= cy <= rect["bottom"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", type=Path, help="JSONL of captures with gold rects and recorded answers")
    parser.add_argument("--presence", type=float, default=0.5, help="uncalibrated presence (noul) threshold")
    parser.add_argument("--box-confidence", type=float, default=0.5, help="uncalibrated box confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a box hit when present")
    args = parser.parse_args(argv)

    presence_correct = presence_total = 0
    box_hits = box_total = 0
    offered_true = offered_false = 0
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row: dict[str, Any] = json.loads(line)
        targets, answers = row["targets"], row.get("answers") or {}
        for key, gold in targets.items():
            present = bool(gold["present"])
            answer = answers.get(key) or {}
            presence = float(answer.get("presence", 0.0))
            predicted_present = presence >= args.presence
            presence_total += 1
            presence_correct += int(predicted_present == present)
            conf = float(answer.get("box_confidence", 0.0))
            box = answer.get("box")
            offered = predicted_present and conf >= args.box_confidence and isinstance(box, dict)
            if present:
                offered_true += int(offered)
            else:
                offered_false += int(offered)
            if not (present and offered and isinstance(box, dict)):
                continue
            box_total += 1
            rect = gold["rect"]
            hit = centre_in_rect(box, rect) or iou(box, rect) >= args.iou
            box_hits += int(hit)

    summary = {
        "presence_accuracy": None if not presence_total else presence_correct / presence_total,
        "presence_n": presence_total,
        "box_hit_rate": None if not box_total else box_hits / box_total,
        "box_n": box_total,
        "false_offers": offered_false,
        "true_offers": offered_true,
        "thresholds": {
            "presence": args.presence,
            "box_confidence": args.box_confidence,
            "iou": args.iou,
            "note": "uncalibrated defaults; tune on your own captures",
        },
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
