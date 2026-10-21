# Test-only launcher: s1a with a 30 s decision limit AND the browser page compressed before every decision, the way
# PR #9 folds it for Laya: url and title, one short line per element, the last 3 actions, no page text.
# The questions and the model are unchanged; only the state the model reads is smaller.
import sys

import s1a.decision_models.jev as jev

LABEL_CHARS, TITLE_CHARS, HISTORY_KEPT = 40, 80, 3
FLAGS = (("checked", "C"), ("selected", "S"), ("expanded", "X"), ("click_did_nothing", "D"))


def row_line(row):
    label = str(row.get("label") or "")[:LABEL_CHARS]
    value = str(row.get("value") or "")[:LABEL_CHARS]
    flags = "".join(letter for key, letter in FLAGS if row.get(key))
    parts = [str(row.get("index", "")), str(row.get("role") or ""), label]
    if value:
        parts.append(f"={value}")
    if flags:
        parts.append(f"[{flags}]")
    if row.get("blocked_by"):
        parts.append(f"(blocked by {str(row['blocked_by'])[:24]})")
    return " ".join(p for p in parts if p)


def compact(state):
    if not (
        isinstance(state, dict) and isinstance(state.get("page"), dict) and isinstance(state.get("elements"), list)
    ):
        return state
    folded = {
        "page": {"url": str(state["page"].get("url", "")), "title": str(state["page"].get("title", ""))[:TITLE_CHARS]},
        "elements": [row_line(row) for row in state["elements"]],
    }
    recent = state.get("recent_actions")
    if recent:
        folded["recent_actions"] = [
            f"{e.get('kind', '')}:{e.get('action', '')}" + ("" if e.get("page_changed") else " (no change)")
            for e in recent[-HISTORY_KEPT:]
        ]
    return folded


_orig_from_env = jev.JevModel.from_env.__func__
jev.JevModel.from_env = classmethod(lambda cls, timeout_s=30.0: _orig_from_env(cls, timeout_s=timeout_s))

_orig_decide = jev.JevModel._decide


async def _decide_compact(self, observation, questions):
    folded = type(observation)(compact(observation.state), images=observation.images)
    return await _orig_decide(self, folded, questions)


jev.JevModel._decide = _decide_compact

from s1a.entry import s1a  # noqa: E402

sys.argv = ["s1a", *sys.argv[1:]]
sys.exit(s1a())
