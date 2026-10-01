# coding: utf-8
"""Compare ticket-router jobs that ran the same seeds with different decision models (in-process vs served Laya).

    uv run python evals/ticket_router/compare_served.py C-in=<job dir> C-direct=<job dir> C-front=<job dir> \
        [--json records.json]

Prints one markdown table (correct routes, per-decision latency p50/p95, total episode time, the answering model and
where it ran), how many (seed, ticket) pairs each two configurations routed the same, and every pair routed
differently with each configuration's top two probabilities. A job dir is what ``s1a run ticket_router`` prints as
``job_dir``; episode time excludes process start and model load. ``--json`` also writes every decision (seed,
ticket, expected and predicted queue, probabilities, ms, model, served_by) per configuration, since job dirs stay local.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Trial:
    seed: int
    router: dict[str, Any]  # the episode's ticket_router extra: ticket_ids, routes, correct, total
    ticks: list[dict[str, Any]]
    elapsed_s: float


def load(job_dir: Path) -> list[Trial]:
    trials = []
    for trial_dir in sorted(path.parent for path in job_dir.glob("*/result.json")):
        episode = json.loads((trial_dir / "agent" / "episode.json").read_text(encoding="utf-8"))
        result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
        router = episode["extra"]["ticket_router"]
        elapsed = float(result["agent_result"]["metadata"]["elapsed_s"])
        trials.append(Trial(router["seed"], router, episode["decisions"], elapsed))
    return sorted(trials, key=lambda trial: trial.seed)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def decided(trial: Trial) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(route, tick) per processed ticket. An episode that stopped early has fewer of both than its batch."""
    pairs = list(zip(trial.router["routes"], trial.ticks, strict=True))
    for route, tick in pairs:
        if route["predicted"] != tick["key"]:
            raise ValueError(
                f"seed {trial.seed}: {route['id']} routed {route['predicted']}, its tick says {tick['key']}"
            )
    return pairs


def routes(trials: list[Trial]) -> dict[tuple[int, str], tuple[str, dict[str, float]]]:
    """(seed, ticket id) -> (predicted queue, probabilities) for every processed ticket."""
    return {
        (trial.seed, route["id"]): (tick["key"], tick["probabilities"])
        for trial in trials
        for route, tick in decided(trial)
    }


def row(name: str, trials: list[Trial]) -> str:
    ticks = [tick for trial in trials for tick in trial.ticks]
    ms = [tick["ms"] for tick in ticks]
    correct = sum(trial.router["correct"] for trial in trials)
    total = sum(trial.router["total"] for trial in trials)
    served_by = ticks[0].get("served_by") or {}
    where = "in process"
    if served_by:
        dtype = (served_by.get("weights_dtype") or "").removeprefix("torch.")
        parts = [
            str(part)
            for part in (served_by.get("device"), dtype, "compiled" if served_by.get("compiled") else "")
            if part
        ]
        where = f"{' '.join(parts) or 'device not reported'} via {ticks[0]['url']}"
    return (
        f"| {name} | {correct}/{total} | {statistics.median(ms):.0f} | {percentile(ms, 0.95):.0f} | "
        f"{sum(trial.elapsed_s for trial in trials):.1f} | {ticks[0].get('model') or ticks[0]['source']} | {where} |"
    )


def records(trials: list[Trial]) -> list[dict[str, Any]]:
    """One row per decision; ``served_by`` and ``url`` only on the first row and where they change."""
    rows, last = [], None
    for trial in trials:
        for route, tick in decided(trial):
            row = {"seed": trial.seed, "ticket": route["id"], "expected": route["expected"]}
            row |= {k: tick.get(k) for k in ("key", "probabilities", "ms", "input_tokens", "model")}
            source = {k: tick[k] for k in ("served_by", "url") if k in tick}
            if source and source != last:
                row |= source
            rows.append(row)
            last = source
    return rows


def main(arguments: list[str]) -> None:
    json_path = None
    if "--json" in arguments:
        at = arguments.index("--json")
        json_path, arguments = Path(arguments[at + 1]), arguments[:at] + arguments[at + 2 :]
    configs = {name: load(Path(path)) for name, _, path in (argument.partition("=") for argument in arguments)}
    if json_path is not None:
        blocks = [
            json.dumps(name) + ": [\n" + ",\n".join(json.dumps(row) for row in records(trials)) + "\n]"
            for name, trials in configs.items()
        ]
        json_path.write_text("{\n" + ",\n".join(blocks) + "\n}\n", encoding="utf-8")  # one decision per line, for diffs
    print("| config | correct | p50 ms | p95 ms | episodes s | model | served on |")
    print("|---|---:|---:|---:|---:|---|---|")
    for name, trials in configs.items():
        print(row(name, trials))
    picked = {name: routes(trials) for name, trials in configs.items()}
    names = list(picked)
    keys = sorted(set.intersection(*(set(found) for found in picked.values())))
    print()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            same = sum(picked[a][key][0] == picked[b][key][0] for key in keys)
            print(f"- {a} and {b} route {same} of {len(keys)} (seed, ticket) pairs the same")
    differing = [key for key in keys if len({picked[name][key][0] for name in names}) > 1]
    if not differing:
        return
    print()
    print("| seed | ticket | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for seed, ticket in differing:
        cells = []
        for name in names:
            key, probabilities = picked[name][(seed, ticket)]
            top = sorted(probabilities.items(), key=lambda item: -item[1])[:2]
            cells.append(f"{key} ({', '.join(f'{k} {p:.3f}' for k, p in top)})")
        print(f"| {seed} | {ticket} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
