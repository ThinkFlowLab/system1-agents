"""Run `s1a` with every Laya call's input and output appended to $LAYA_DUMP (one JSON line per call)."""

import json
import os
import sys

import laya.agent

_orig = laya.agent.Agent.system_one


def _dumped(self, state, questions, *args, **kwargs):
    result = _orig(self, state, questions, *args, **kwargs)
    with open(os.environ["LAYA_DUMP"], "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"state": state, "questions": questions, "result": result}, ensure_ascii=False, default=str)
            + "\n"
        )
    return result


laya.agent.Agent.system_one = _dumped

from s1a.entry import s1a  # noqa: E402

sys.argv = ["s1a", *sys.argv[1:]]
sys.exit(s1a())
