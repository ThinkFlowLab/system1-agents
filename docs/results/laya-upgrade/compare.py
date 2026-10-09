# coding: utf-8
"""Tables from the outputs of laya_ab.py:  python compare.py records/

For each checkpoint and device: load time and latency per run for baseline and head, and whether every
answer of every run equals the first baseline run's. Then every run with LAYA_MPS_AMP_MIN_ROWS set, against
the baseline on the same device: the largest change in any probability and each decision that changed.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ANSWERS = ("one_question", "one_noul", "six_questions")


def answers(record: dict) -> list:
    return [{key: ticket[key] for key in ANSWERS} for ticket in record["tickets"]]


def changes(reference: dict, record: dict) -> tuple[float, list[str]]:
    largest, changed = 0.0, []
    for a, b in zip(reference["tickets"], record["tickets"], strict=True):
        for part in ANSWERS:
            for name, value in a[part].items():
                other = b[part][name]
                if name == "probabilities":
                    largest = max([largest] + [abs(p - other[option]) for option, p in value.items()])
                elif name == "pick":
                    if value != other:
                        changed.append(f"{a['id']} {part} pick: {value} -> {other}")
                elif name != "confidence":
                    largest = max(largest, abs(value - other))
                    if (value >= 0.5) != (other >= 0.5):
                        changed.append(f"{a['id']} {part} {name}: {value} -> {other}")
    return round(largest, 4), changed


def main(folder: str) -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(folder).glob("*.json"))]
    cells: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        kind = (
            "no-skip"
            if record["no_skip"]
            else "override"
            if "LAYA_MPS_AMP_MIN_ROWS" in record["environment"]
            else record["label"]
        )
        cells[(record["checkpoint"], record["device"])][kind].append(record)
    for record in records[:1]:
        print(f"{record['chip']}, {record['os']}, torch {record['torch']}, transformers {record['transformers']}")
    for label in ("baseline", "head"):
        seen = {(r["s1a_commit"][:7], r["laya"], r["checkpoint_revision"][:7]) for r in records if r["label"] == label}
        print(f"{label}: " + "; ".join(f"s1a {c}, laya {v}, checkpoint revision {rev}" for c, v, rev in sorted(seen)))
    print()
    print(
        "| checkpoint | device | load s, baseline | load s, head | one question p50 ms, baseline | head | six questions p50 ms, baseline | head | answers |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for (checkpoint, device), kinds in sorted(cells.items()):
        base, head = kinds["baseline"], kinds["head"]
        same = all(answers(r) == answers(base[0]) for r in base + head)
        column = lambda rows, key: " / ".join(str(r[key]) for r in rows)  # noqa: E731
        print(
            f"| {checkpoint} | {device} | {column(base, 'load_s')} | {column(head, 'load_s')} | "
            f"{column(base, 'one_question_p50_ms')} | {column(head, 'one_question_p50_ms')} | "
            f"{column(base, 'six_questions_p50_ms')} | {column(head, 'six_questions_p50_ms')} | "
            f"{'identical' if same else 'DIFFERENT'} ({len(base)} + {len(head)} runs) |"
        )
    for (checkpoint, device), kinds in sorted(cells.items()):
        for record in kinds["override"]:
            largest, changed = changes(kinds["baseline"][0], record)
            setting = record["environment"]["LAYA_MPS_AMP_MIN_ROWS"]
            print(f"\n{checkpoint} on {device}, head with LAYA_MPS_AMP_MIN_ROWS={setting}, against baseline:")
            print(f"  largest probability change {largest}; decisions changed: {len(changed)}")
            for line in changed:
                print("  " + line)
        for record in kinds["no-skip"]:
            print(f"\n{checkpoint} on {device}, baseline without its weight-init skip: load {record['load_s']} s")


main(sys.argv[1])
