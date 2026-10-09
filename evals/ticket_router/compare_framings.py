"""Score the ticket-router set under several input framings, against one live clm-serve.

The tool loop sends one shape; CLM's own examples send another. This sends the same tickets, the same five queue
descriptions and the same rules under each, so the difference between the rows is the framing and nothing else.

    CLM_URL=http://127.0.0.1:8091 python evals/ticket_router/compare_framings.py --out framings.json

Each framing is one decision per (seed, ticket); seeds 0, 1 and 2 shuffle the same 30 tickets, which is what
`s1a run ticket_router --episodes 3 --seed 0` does.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

from s1a.agents.ticket_router import PUBLIC_FIELDS, QUEUES, RULES, load_tickets

TICKETS = Path(__file__).resolve().parents[2] / "s1a" / "agents" / "_data" / "ticket_router_eval.jsonl"
SHORT_QUESTION = "Which queue does this ticket belong to?"


def observation(row: dict) -> dict:
    return {"ticket": {k: row[k] for k in PUBLIC_FIELDS if k in row}, "progress": {"ticket_id": row["id"]}}


def ticket_text(row: dict) -> str:
    """The ticket as one sentence, the shape CLM's own examples use for a state."""
    parts = [f"Ticket: {row['title']}. {row['description']}"]
    if row.get("order_status"):
        parts.append(f"Order status: {row['order_status']}.")
    return " ".join(parts)


def cua_style(row: dict) -> str:
    """A context the way the cua backend composes one: a header, the state, then the rules as context."""
    return "\n".join(["# s1a / ticket_router", ticket_text(row), RULES])


FRAMINGS = {
    # what the tool loop sends today: the observation as JSON, the rules as the question
    "agent": lambda row: (observation(row), RULES),
    # the same information, the rules on the context side and a short question, as cua_context does
    "cua-style": lambda row: (cua_style(row), SHORT_QUESTION),
    # what this backend could compose without inventing any text: a header, the state, the rules, no question.
    # cua composes its context this way; the loop gives ticket_router no goal, so nothing else can go in the header.
    "cua-no-question": lambda row: ("\n".join(["# s1a / ticket_router", ticket_text(row), RULES]), None),
    # the ticket alone: the state shape CLM's own examples use
    "short": lambda row: (ticket_text(row), SHORT_QUESTION),
    # the shortest thing that still names the task
    "short-no-rules": lambda row: (ticket_text(row), None),
}


def ask(url: str, model: str, state: object, instructions: str | None, timeout: float) -> str:
    body = {
        "model": model,
        "state": state,
        "questions": {"pick": {"type": "choice", "instructions": instructions, "criteria": QUEUES}},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["answers"]["pick"]["choice"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="clm-latest")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_tickets(TICKETS)
    batches = []
    for seed in args.seeds:
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        batches.append((seed, shuffled[: args.batch_size]))

    report: dict = {"url": args.url, "model": args.model, "seeds": args.seeds, "framings": {}}
    for name, build in FRAMINGS.items():
        hits, per_seed, wrong = 0, {}, []
        for seed, batch in batches:
            seed_hits = 0
            for row in batch:
                state, instructions = build(row)
                got = ask(args.url, args.model, state, instructions, args.timeout)
                seed_hits += got == row["label"]
                if got != row["label"]:
                    wrong.append({"seed": seed, "id": row["id"], "expected": row["label"], "got": got})
            per_seed[seed] = seed_hits
            hits += seed_hits
        total = sum(len(b) for _, b in batches)
        answers = [w["got"] for w in wrong]
        report["framings"][name] = {
            "correct": hits,
            "total": total,
            "accuracy": round(hits / total, 4),
            "per_seed": per_seed,
            "answers_given": sorted({w["got"] for w in wrong} | {r["label"] for r in rows}),
            "distinct_wrong_answers": sorted(set(answers)),
            "wrong": wrong,
        }
        print(f"{name:16} {hits:3}/{total} = {hits / total:6.1%}   per seed {per_seed}", flush=True)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
