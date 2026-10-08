# coding: utf-8
"""Episodes, summaries and the Harbor-shaped job folders that Nullius opens."""

from __future__ import annotations

import json
import random
import shutil
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from s1a.config import HOME

RESULTS_DIR = HOME / "evals" / "results"
SHOWCASE_DIR = HOME / "evals" / "showcase"  # --showcase runs: frames and replays, never read by evals.table
BOOTSTRAP_RESAMPLES = 1000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Episode:
    env: str
    policy: str
    seed: int
    score: float
    steps: int
    elapsed_s: float
    started_at: str
    finished_at: str
    final_state: dict[str, Any]
    chat_calls: int
    chat_input_tokens: int
    chat_output_tokens: int
    chat_cache_tokens: int  # the part of chat_input_tokens the provider served from its prompt cache
    jev_input_tokens: int
    invalid_keys: int
    cost_usd: float | None
    error: str | None = None  # why the model could not play the episode; None for a played one
    decisions: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    views: list[dict[str, Any]] = field(default_factory=list)  # views[i] is what act i chose from; the last is final
    frames_dir: Path | None = None  # PNGs taken during a --showcase run, moved into the trial folder by write_job


def bootstrap_interval(values: list[float], *, resamples: int, seed: int) -> tuple[float, float]:
    """95 % percentile bootstrap of the mean."""
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return round(means[int(0.025 * resamples)], 3), round(means[int(0.975 * resamples) - 1], 3)


def summarize(episodes: list[Episode]) -> dict[str, Any]:
    """Score statistics over the episodes that played (``error`` None); counts and totals over all of them."""
    scored = [episode for episode in episodes if episode.error is None]
    scores = [episode.score for episode in scored]
    latencies = [decision["ms"] for episode in episodes for decision in episode.decisions if "ms" in decision]
    costs = [episode.cost_usd for episode in episodes]
    return {
        "env": episodes[0].env,
        "policy": episodes[0].policy,
        "episodes": len(episodes),
        "scored": len(scored),
        "mean_score": round(statistics.mean(scores), 3) if scores else None,
        "mean_score_ci95": list(bootstrap_interval(scores, resamples=BOOTSTRAP_RESAMPLES, seed=0)) if scores else None,
        "wins": sum(score > 0 for score in scores),
        "losses": sum(score < 0 for score in scores),
        "mean_steps": round(statistics.mean(episode.steps for episode in scored), 1) if scored else None,
        "mean_elapsed_s": round(statistics.mean(episode.elapsed_s for episode in episodes), 1),
        "decisions": sum(len(episode.decisions) for episode in episodes),
        "median_decision_ms": int(statistics.median(latencies)) if latencies else 0,
        "chat_calls": sum(episode.chat_calls for episode in episodes),
        "jev_input_tokens": sum(episode.jev_input_tokens for episode in episodes),
        "chat_input_tokens": sum(episode.chat_input_tokens for episode in episodes),
        "chat_output_tokens": sum(episode.chat_output_tokens for episode in episodes),
        "chat_cache_tokens": sum(episode.chat_cache_tokens for episode in episodes),
        "invalid_keys": sum(episode.invalid_keys for episode in episodes),
        "cost_usd": None if None in costs else round(sum(cost for cost in costs if cost is not None), 6),
        "rethinks": sum(len(episode.extra.get("rethinks") or []) for episode in episodes),
        "errors": sum(episode.error is not None for episode in episodes),
        "total_s": round(sum(episode.elapsed_s for episode in episodes), 1),
    }


def write_job(eval_name: str, episodes: list[Episode], *, results_dir: Path) -> Path:
    """One Harbor job folder per run: config.json, result.json, summary.json, one trial folder per episode.

    The trial shape follows agent-eval's runner so Nullius reads both roots alike.
    """
    policy = episodes[0].policy
    agent_name = f"s1a-evals/{policy}"
    job_dir = (
        results_dir / eval_name / f"{datetime.now():%Y-%m-%d__%H-%M-%S-%f}__{policy}"
    )  # microseconds: two jobs in one second
    job_dir.mkdir(parents=True)
    for index, episode in enumerate(episodes):
        trial_dir = job_dir / f"{eval_name}--{index}__{uuid.uuid4().hex[:7]}"
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "agent" / "episode.json").write_text(
            json.dumps(
                {
                    "decisions": episode.decisions,
                    "final_state": episode.final_state,
                    "extra": episode.extra,
                    "views": episode.views,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if episode.frames_dir is not None:
            shutil.move(str(episode.frames_dir), str(trial_dir / "agent" / "frames"))
        trial = {
            "id": str(uuid.uuid4()),
            "task_name": f"{eval_name}/{episode.seed}",
            "trial_name": trial_dir.name,
            "task_id": {"path": None},
            "source": eval_name,
            "agent_info": {
                "name": agent_name,
                "version": None,
                "model_info": {"name": policy, "provider": "s1a"},
            },
            "agent_result": {
                "n_input_tokens": episode.jev_input_tokens + episode.chat_input_tokens,
                "n_cache_tokens": episode.chat_cache_tokens,
                "n_output_tokens": episode.chat_output_tokens,
                "cost_usd": episode.cost_usd,
                "metadata": {
                    "steps": episode.steps,
                    "elapsed_s": episode.elapsed_s,
                    "decisions": len(episode.decisions),
                    "chat_calls": episode.chat_calls,
                    "jev_input_tokens": episode.jev_input_tokens,
                    "invalid_keys": episode.invalid_keys,
                },
            },
            "verifier_result": {"rewards": {"reward": episode.score}},
            "exception_info": {"error": episode.error} if episode.error is not None else None,
            "started_at": episode.started_at,
            "finished_at": episode.finished_at,
            "agent_execution": {"started_at": episode.started_at, "finished_at": episode.finished_at},
        }
        (trial_dir / "result.json").write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")
    finished_at = now_iso()
    errored = sum(episode.error is not None for episode in episodes)
    job = {
        "id": str(uuid.uuid4()),
        "started_at": episodes[0].started_at,
        "updated_at": finished_at,
        "finished_at": finished_at,
        "n_total_trials": len(episodes),
        "stats": {
            "n_completed_trials": len(episodes) - errored,
            "n_errored_trials": errored,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
        },
    }
    (job_dir / "result.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    config = {
        "job_name": job_dir.name,
        "agents": [{"name": agent_name, "model_name": policy}],
        "datasets": [{"name": eval_name, "n_tasks": len(episodes)}],  # the dict shape Nullius's parse_job reads
    }
    (job_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (job_dir / "summary.json").write_text(json.dumps(summarize(episodes), indent=2), encoding="utf-8")
    return job_dir
