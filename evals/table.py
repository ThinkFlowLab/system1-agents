# coding: utf-8
"""One table over every job folder: scored outcomes and all-attempt operational totals.

``python -m evals.table evals/results`` reads the job folders that ``write_job`` produces, one ``result.json`` per
trial. Rows are keyed by the results root's child folder. Failed trials stay in all-attempt counts, timing, and cost
when those values were recorded; only explicitly present rewards from non-error trials enter score statistics.
"""

from __future__ import annotations

import s1a.entry  # noqa: F401  # routes the harness logs to files before anything imports openjiuwen
import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from s1a.jobs import BOOTSTRAP_RESAMPLES, bootstrap_interval

COLUMNS = (
    "eval",
    "model",
    "attempts",
    "errors",
    "N",
    "score (n)",
    "score unknown",
    "score",
    "scored median s",
    "steps",
    "decisions",
    "chat calls",
    "scored $ / episode",
    "time (n/attempts)",
    "s / attempt",
    "time unknown",
    "cost (n/attempts)",
    "$ / attempt",
    "cost unknown",
)


@dataclass(frozen=True)
class Trial:
    eval_name: str
    model: str
    errored: bool  # result.json holds exception_info: no score, counted in the errors column only
    score: float | None
    elapsed_s: float | None
    steps: int | None  # None when the runner recorded no step count
    decisions: int
    chat_calls: int
    cost_usd: float | None


def window_s(result: dict[str, Any]) -> float | None:
    """Seconds between a trial's recorded start and finish."""
    window = result.get("agent_execution") or {}
    started, finished = window.get("started_at"), window.get("finished_at")
    if not started or not finished:
        return None
    return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()


def model_label(result: dict[str, Any]) -> str:
    """The model a trial's ``result.json`` names: ``jev``, ``llm``, ``random`` or a rule name, after any ``<suite>/`` prefix."""
    name = str((result.get("agent_info") or {}).get("name") or "")
    return name.partition("/")[2] or name


def read_trial(trial_dir: Path, *, eval_name: str) -> Trial | None:
    """One trial from its ``result.json``."""
    result_file = trial_dir / "result.json"
    if not result_file.is_file():
        return None
    result = json.loads(result_file.read_text(encoding="utf-8"))
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    steps = metadata.get("steps")
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    elapsed = metadata.get("elapsed_s")
    if elapsed is None:
        elapsed = window_s(result)
    return Trial(
        eval_name=eval_name,
        model=model_label(result),
        errored=result.get("exception_info") is not None,
        score=float(reward) if reward is not None else None,
        elapsed_s=float(elapsed) if elapsed is not None else None,
        steps=int(steps) if steps is not None else None,
        decisions=int(metadata.get("decisions") or 0),
        chat_calls=int(metadata.get("chat_calls") or 0),
        cost_usd=agent_result.get("cost_usd"),
    )


def read_results(root: Path) -> list[Trial]:
    """Every trial under ``root/<eval>/<job>/<trial>/result.json``, named after the eval folder."""
    trials = []
    for result_file in sorted(root.glob("*/*/*/result.json")):
        trial = read_trial(result_file.parent, eval_name=result_file.relative_to(root).parts[0])
        if trial is not None:
            trials.append(trial)
    return trials


def rows(trials: list[Trial]) -> list[dict[str, Any]]:
    """One row per (eval, model). Score uses non-error trials with a recorded reward; time and cost use every
    attempt that recorded that value. Each count is exposed so missing values never silently become zero."""
    groups: dict[tuple[str, str], list[Trial]] = {}
    for trial in trials:
        groups.setdefault((trial.eval_name, trial.model), []).append(trial)
    table = []
    for (eval_name, model), all_members in sorted(groups.items()):
        members = [member for member in all_members if not member.errored]
        scores = [member.score for member in members if member.score is not None]
        scored_elapsed = [member.elapsed_s for member in members if member.elapsed_s is not None]
        elapsed = [member.elapsed_s for member in all_members if member.elapsed_s is not None]
        scored_costs = [member.cost_usd for member in members]
        costs = [member.cost_usd for member in all_members if member.cost_usd is not None]
        steps = [member.steps for member in members if member.steps is not None]
        table.append(
            {
                "eval": eval_name,
                "model": model,
                "N": len(members),
                "errors": len(all_members) - len(members),
                "attempts": len(all_members),
                "score_n": len(scores),
                "score_unknown": len(members) - len(scores),
                "mean_score": round(statistics.mean(scores), 3) if scores else None,
                "ci95": bootstrap_interval(scores, resamples=BOOTSTRAP_RESAMPLES, seed=0) if scores else None,
                "time_n": len(elapsed),
                "mean_elapsed_all_s": round(statistics.mean(elapsed), 1) if elapsed else None,
                "time_unknown": len(all_members) - len(elapsed),
                "median_s": round(statistics.median(scored_elapsed), 1) if scored_elapsed else None,
                "mean_steps": round(statistics.mean(steps), 1) if steps else None,
                "mean_decisions": round(statistics.mean(member.decisions for member in members), 1)
                if members
                else None,
                "mean_chat_calls": round(statistics.mean(member.chat_calls for member in members), 1)
                if members
                else None,
                "mean_cost_usd": round(statistics.mean(scored_costs), 4)
                if scored_costs and None not in scored_costs
                else None,
                "cost_n": len(costs),
                "mean_cost_all_usd": round(statistics.mean(costs), 4) if costs else None,
                "cost_unknown": len(all_members) - len(costs),
            }
        )
    return table


def markdown(table: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in table:
        cost = "n/a" if row["mean_cost_all_usd"] is None else f"{row['mean_cost_all_usd']:.4f}"
        scored_cost = "n/a" if row["mean_cost_usd"] is None else f"{row['mean_cost_usd']:.4f}"
        score = "n/a" if row["ci95"] is None else f"{row['mean_score']} [{row['ci95'][0]}, {row['ci95'][1]}]"
        cells = [
            row[key] if row[key] is not None else "n/a"
            for key in ("median_s", "mean_steps", "mean_decisions", "mean_chat_calls")
        ]
        lines.append(
            f"| {row['eval']} | {row['model']} | {row['attempts']} | {row['errors']} | {row['N']} | "
            f"{row['score_n']}/{row['attempts']} | {row['score_unknown']} | {score} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {scored_cost} | "
            f"{row['time_n']}/{row['attempts']} | {row['mean_elapsed_all_s'] if row['mean_elapsed_all_s'] is not None else 'n/a'} | "
            f"{row['time_unknown']} | {row['cost_n']}/{row['attempts']} | {cost} | {row['cost_unknown']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="the results root, e.g. evals/results")
    args = parser.parse_args()
    print(markdown(rows(read_results(args.root))))


if __name__ == "__main__":
    main()
