"""The recorded Flights ticks shaped exactly as LayaModel._decide shapes them, one /v1/systemone body per tick.

uv run python docs/results/laya-browser-shaping/served-system1-omni/make_bodies.py
"""

import json
from pathlib import Path

from s1a.decision_models.laya import laya_browser_question, laya_state

HERE = Path(__file__).resolve().parent
with open(HERE / "requests.jsonl", "w", encoding="utf-8") as out:
    for i, line in enumerate(open(HERE.parent / "zurich-london.jsonl", encoding="utf-8")):
        tick = json.loads(line)
        questions = {name: laya_browser_question(q, name) for name, q in tick["questions"].items()}
        out.write(
            json.dumps(
                {"tick": i, "state": laya_state(tick["state"]), "questions": questions, "jev": tick["answers"]},
                ensure_ascii=False,
            )
            + "\n"
        )
