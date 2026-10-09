"""Served Laya through system1-omni's Rust frontend on the recorded Flights ticks, PR #9's shaping applied.

Each tick is sent three ways with byte-identical state and questions:
  served-1536: POST frontend /v1/systemone with max_len=1536, head_max_len=1024 (the PR's browser window)
  served-default: the same body without window fields (the worker's checkpoint window)
  inproc-1536: laya.load() in this process, system_one(..., max_len=1536, head_max_len=1024) when supported
and writes one JSON line per tick and question.
"""

import json
import sys
import time
import urllib.request

FRONTEND = sys.argv[1]
BODIES, OUT = sys.argv[2], sys.argv[3]


def post(body):
    req = urllib.request.Request(
        FRONTEND + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data, round((time.perf_counter() - t0) * 1000)


def pick(answers, name):
    a = (answers or {}).get(name)
    if isinstance(a, dict):
        return a.get("answer", a.get("choice")), a.get("confidence")
    return a, None


import laya  # noqa: E402

agent = laya.load("convaiinnovations/laya")
print(
    "laya",
    laya.__version__,
    "cfg max_len",
    agent.cfg.get("max_len"),
    "head_max_len",
    agent.cfg.get("head_max_len"),
    flush=True,
)
out = open(OUT, "w", encoding="utf-8")
for line in open(BODIES, encoding="utf-8"):
    t = json.loads(line)
    base = {"state": t["state"], "questions": t["questions"]}
    rows = {}
    for label, extra in (("served-1536", {"max_len": 1536, "head_max_len": 1024}), ("served-default", {})):
        try:
            data, ms = post({**base, **extra})
            rows[label] = {
                "answers": data.get("answers"),
                "usage": data.get("usage"),
                "ms": ms,
                "model": data.get("model"),
            }
        except Exception as e:  # keep failures in the evidence
            rows[label] = {"error": f"{type(e).__name__}: {e}"}
    t0 = time.perf_counter()
    try:
        try:
            res = agent.system_one(t["state"], t["questions"], max_len=1536, head_max_len=1024)
            how = "kwargs"
        except TypeError:
            agent.cfg["max_len"], agent.cfg["head_max_len"] = 1536, 1024
            res = agent.system_one(t["state"], t["questions"])
            how = "cfg"
        rows["inproc-1536"] = {
            "answers": res.get("answers"),
            "usage": res.get("usage"),
            "ms": round((time.perf_counter() - t0) * 1000),
            "how": how,
        }
    except Exception as e:
        rows["inproc-1536"] = {"error": f"{type(e).__name__}: {e}"}
    for name in t["questions"]:
        rec = {"tick": t["tick"], "question": name, "jev": t["jev"].get(name)}
        for label, r in rows.items():
            rec[label] = r.get("error") or pick(r.get("answers"), name)[0]
            rec[label + "_conf"] = None if "error" in r else pick(r.get("answers"), name)[1]
        rec["input_tokens"] = {
            k: (v.get("usage") or {}).get("input_tokens") for k, v in rows.items() if "error" not in v
        }
        rec["ms"] = {k: v.get("ms") for k, v in rows.items()}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
    print("tick", t["tick"], {k: ("ERR" if "error" in v else v.get("ms")) for k, v in rows.items()}, flush=True)
print("TRIAL_DONE", flush=True)
