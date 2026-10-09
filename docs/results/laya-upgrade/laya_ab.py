# coding: utf-8
"""One run of `--model laya`'s backend in a fresh process; run.sh calls it from each checkout's root.

It loads the model through `LayaModel.from_env()`, then for each of the 30 tickets in the repository's
ticket-router file asks one choice question, one noul question, and one request of six questions. The JSON
it writes holds every answer, the timings, and what ran: commit, library versions, checkpoint revision,
device and the Laya variables that were set. `--no-skip` loads without main's weight-init skip.
"""

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path

T0 = time.perf_counter()
NOULS = {
    "refund": "Does the customer ask for a refund?",
    "angry": "Is the customer angry?",
    "cancel": "Does the customer want to cancel the order?",
    "human": "Does the customer ask for a human?",
    "late": "Is a delivery late?",
}
LAYA_VARIABLES = (
    "LAYA_MODEL",
    "LAYA_SUBFOLDER",
    "LAYA_DEVICE",
    "LAYA_MAX_LEN",
    "LAYA_HEAD_MAX_LEN",
    "LAYA_MPS_AMP_MIN_ROWS",
)


def run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def checkpoint_revision(repo: str) -> str:
    """The commit the local Hugging Face cache has for the repository's main."""
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else ""


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="baseline or head")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-skip", action="store_true", help="baseline only: load without its weight-init skip")
    args = parser.parse_args()

    from s1a.agents.ticket_router import DEFAULT_DATASET, PUBLIC_FIELDS, QUEUES, RULES, load_tickets
    from s1a.decision_models import ChoiceQuestion, NoulQuestion, Observation
    from s1a.decision_models import laya as laya_module

    if args.no_skip:
        import contextlib

        laya_module.without_weight_init = contextlib.nullcontext
    model = laya_module.LayaModel.from_env()
    load_s = time.perf_counter() - T0

    pick = ChoiceQuestion(dict(QUEUES), rules=RULES)
    tickets, one_ms, six_ms = [], [], []
    for row in load_tickets(DEFAULT_DATASET):
        observation = Observation({key: row[key] for key in PUBLIC_FIELDS if key in row})
        started = time.perf_counter()
        one = await model.decide_many(observation, {"pick": pick})
        one_ms.append((time.perf_counter() - started) * 1000)
        noul = await model.decide_many(observation, {"refund": NoulQuestion(NOULS["refund"])})
        started = time.perf_counter()
        six = await model.decide_many(observation, {"pick": pick, **{k: NoulQuestion(v) for k, v in NOULS.items()}})
        six_ms.append((time.perf_counter() - started) * 1000)
        choice, six_choice = one.choice("pick"), six.choice("pick")
        tickets.append(
            {
                "id": row["id"],
                "one_question": {
                    "pick": choice.key,
                    "probabilities": choice.probabilities,
                    "confidence": choice.confidence,
                },
                "one_noul": {"refund": noul.noul("refund").p},
                "six_questions": {
                    "pick": six_choice.key,
                    "probabilities": six_choice.probabilities,
                    "confidence": six_choice.confidence,
                    **{k: six.noul(k).p for k in NOULS},
                },
            }
        )
    agent = model._agent
    repo = os.getenv("LAYA_MODEL") or laya_module.LAYA_DEFAULT_MODEL
    record = {
        "label": args.label,
        "no_skip": args.no_skip,
        "s1a_commit": run("git", "rev-parse", "HEAD"),
        "laya": metadata.version("laya"),
        "torch": metadata.version("torch"),
        "transformers": metadata.version("transformers"),
        "python": platform.python_version(),
        "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
        "os": "macOS " + platform.mac_ver()[0],
        "checkpoint": model.model,
        "checkpoint_revision": checkpoint_revision(repo),
        "device": str(agent.device),
        "mps_amp_min_rows": getattr(agent, "mps_amp_min_rows", None),
        "environment": {name: os.environ[name] for name in LAYA_VARIABLES if os.getenv(name)},
        "load_average_1m": round(os.getloadavg()[0], 1),
        "load_s": round(load_s, 1),
        "one_question_p50_ms": round(statistics.median(one_ms[1:])),
        "six_questions_p50_ms": round(statistics.median(six_ms[1:])),
        "tickets": tickets,
    }
    head = {key: value for key, value in record.items() if key != "tickets"}
    lines = [json.dumps(head)[:-1] + ', "tickets": ['] + [
        "  " + json.dumps(ticket) + ("," if n < len(tickets) - 1 else "") for n, ticket in enumerate(tickets)
    ]
    Path(args.out).write_text("\n".join(lines) + "\n]}\n", encoding="utf-8")  # one ticket per line
    print(args.out, record["laya"], record["device"], "load", record["load_s"], "s")


asyncio.run(main())
