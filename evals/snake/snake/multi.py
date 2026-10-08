"""Many snakes, one backend: N concurrent games driven by a single worker.

Each game advances at its own paced fps; decisions are issued concurrently
through a thread pool (the client side of a SerialScheduler worker queues
naturally). Frames and a global summary are recorded for rendering.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .game import SnakeGame
from .policy import LayaPolicy
from .warmup import warm_up


class GameRunner:
    def __init__(self, policy, seed, width, height, initial_length):
        self.policy = policy
        self.game = SnakeGame(width, height, seed, initial_length)
        self.seed = seed
        self.steps = 0
        self.deaths = 0
        self.interventions = 0
        self.latest_decision = {}
        self.inference = []
        self.round = 1

    def tick(self):
        if not self.game.alive or self.game.won:
            return False
        decision = self.policy.decide(self.game)
        self.latest_decision = decision.to_dict()
        self.inference.append(decision.inference_ms)
        self.interventions += decision.intervened
        self.game.step(decision.executed)
        self.steps += 1
        return True


def positive(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return result


def run_multi(argv=None):
    parser = argparse.ArgumentParser(prog="laya-snake multi", description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="english")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--fps", type=positive, default=12, help="Per-game decision rate")
    parser.add_argument("--steps", type=int, default=300, help="Per-game step budget")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.games < 1 or args.cols < 1:
        parser.error("--games and --cols must be positive")

    policy = LayaPolicy(args.backend, model=args.model, prompt=args.prompt)
    runners = [
        GameRunner(policy, args.seed + i, args.width, args.height, args.initial_length) for i in range(args.games)
    ]
    warm_up(policy, SnakeGame(args.width, args.height, args.seed + 90000, args.initial_length), steps=4)

    args.record.parent.mkdir(parents=True, exist_ok=True)
    record = args.record.open("x")
    record.write(
        json.dumps(
            {
                "type": "metadata",
                "format": "laya-snake-multi-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": policy.metadata,
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }
        )
        + "\n"
    )
    out_lock = threading.Lock()
    started = time.perf_counter()
    stop = threading.Event()

    def loop(runner, period):
        while not stop.is_set():
            t0 = time.perf_counter()
            progressed = runner.tick()
            shown = time.perf_counter()
            with out_lock:
                record.write(
                    json.dumps(
                        {
                            "type": "frame",
                            "at": shown - started,
                            "seed": runner.seed,
                            "round": runner.round,
                            "game": runner.game.snapshot(),
                            "decision": runner.latest_decision,
                            "steps": runner.steps,
                            "deaths": runner.deaths,
                            "interventions": runner.interventions,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            if not progressed:
                runner.deaths += not runner.game.won
                runner.round += 1
                runner.game = SnakeGame(args.width, args.height, runner.seed + runner.round - 1, args.initial_length)
                runner.latest_decision = {}
            if runner.steps >= args.steps:
                return
            remaining = period - (time.perf_counter() - t0)
            if remaining > 0:
                stop.wait(remaining)

    period = 1 / args.fps
    try:
        with ThreadPoolExecutor(max_workers=args.games) as pool:
            try:
                futures = [pool.submit(loop, r, period) for r in runners]
                for future in as_completed(futures):
                    future.result()
            finally:
                # Stop sibling games before the executor waits for them, including on Ctrl-C.
                stop.set()
        elapsed = time.perf_counter() - started
        all_inference = [ms for r in runners for ms in r.inference]
        summary = {
            "games": args.games,
            "steps_per_game": args.steps,
            "total_steps": sum(r.steps for r in runners),
            "seconds": elapsed,
            "decisions_per_second": sum(r.steps for r in runners) / elapsed,
            "mean_inference_ms": sum(all_inference) / len(all_inference) if all_inference else None,
            "deaths": sum(r.deaths for r in runners),
            "interventions": sum(r.interventions for r in runners),
            "scores": [r.game.score for r in runners],
            "best_score": max((r.game.score for r in runners), default=0),
        }
        record.write(json.dumps({"type": "end", "summary": summary}) + "\n")
    finally:
        record.close()
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0
