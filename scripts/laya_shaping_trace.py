"""What Laya receives on the browser front, with the compaction on and off, on the same recorded input.

Reads a recording of browser ticks (one JSON object per line: ``state`` and ``questions`` exactly as the browser
front sent them to Jev, ``answers`` the reference answer) and prints, per tick, the token counts Laya's own
``build_sequence`` produces with ``LAYA_COMPACT_BROWSER_STATE`` off and on, under the checkpoint's default window
(512 / 192) and the browser window (1536 / 1024). Only Laya's tokenizer is loaded, not its weights: the counts are
what the model would read, no forward pass is run.

    uv run --extra laya python scripts/laya_shaping_trace.py docs/results/laya-browser-shaping/zurich-london.jsonl

``--show TICK`` also prints that tick's target options as Laya reads them, both ways.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from s1a.decision_models.laya import LAYA_DEFAULT_MODEL, laya_browser_question, laya_state

WINDOWS = ((512, 192), (1536, 1024))  # the checkpoint's default; LAYA_MAX_LEN / LAYA_HEAD_MAX_LEN for browser runs


def load_tokenizer(model: str) -> Any:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(model, allow_patterns=["tokenizer/*", "rl_agent_config.json"])
    return AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer")), os.path.basename(path)


def measure(tok: Any, state: Any, asked: dict[str, Any], max_len: int, head_max_len: int) -> dict[str, Any]:
    """Laya's own sequence for one question: how many tokens the options take, whether they were cut to an equal
    share, and how much of the state fits the window."""
    from laya.agent import Agent
    from laya.common import build_sequence, render_options, serialize_state

    q = Agent._to_internal(asked)
    options = render_options(q)
    full = [len(tok(" " + o, add_special_tokens=False)["input_ids"][:48]) + 1 for o in options]
    cut = sum(full) > head_max_len - 16
    _, markers = build_sequence(tok, state, q, max_len, head_max_len)
    head_only, _ = build_sequence(tok, "", q, max_len, head_max_len)  # [CLS] instruction [SEP] options [SEP] [SEP]
    state_tokens = len(tok(serialize_state(state), add_special_tokens=False)["input_ids"])
    return {
        "options": len(options),
        "option_tokens": sum(full),
        "cut": cut,
        "per_option": max(4, (head_max_len - 16) // max(1, len(options))) if cut else None,
        "state_tokens": state_tokens,
        "state_kept": max(0, min(state_tokens, max_len - len(head_only))),
        "all_options_kept": len(markers) == len(options),
    }


def shaped(state: Any, questions: dict[str, Any], compact: bool) -> tuple[Any, dict[str, Any]]:
    if not compact:
        return state, questions
    return laya_state(state), {name: laya_browser_question(asked, name) for name, asked in questions.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording")
    parser.add_argument("--show", type=int, action="append", default=[], help="print this tick's target options")
    parser.add_argument("--model", default=os.getenv("LAYA_MODEL", LAYA_DEFAULT_MODEL))
    args = parser.parse_args()

    tok, revision = load_tokenizer(args.model)
    ticks = [json.loads(line) for line in open(args.recording, encoding="utf-8") if line.strip()]
    print(f"recording: {args.recording} ({len(ticks)} ticks)  tokenizer: {args.model}@{revision}")
    for max_len, head_max_len in WINDOWS:
        print(f"\n== window max_len={max_len} head_max_len={head_max_len}")
        print("tick elements question          | off: state tok, options tok, cut  | on: state tok, options tok, cut")
        for i, tick in enumerate(ticks):
            off_state, off_q = shaped(tick["state"], tick["questions"], compact=False)
            on_state, on_q = shaped(tick["state"], tick["questions"], compact=True)
            for name in tick["questions"]:
                off = measure(tok, off_state, off_q[name], max_len, head_max_len)
                on = measure(tok, on_state, on_q[name], max_len, head_max_len)

                def cell(m: dict[str, Any]) -> str:
                    cut = f"cut to {m['per_option']}/option" if m["cut"] else "fits"
                    return f"{m['state_tokens']:>6} ({m['state_kept']:>4} kept) {m['option_tokens']:>5}  {cut:<17}"

                elements = len(tick["state"].get("elements", []))
                print(f"{i:>4} {elements:>8} {name:<18}| {cell(off)} | {cell(on)}")

    for i in args.show:
        tick = ticks[i]
        for compact in (False, True):
            _, questions = shaped(tick["state"], tick["questions"], compact)
            target = next((n for n in questions if n.endswith("_target") and n.startswith("click")), None)
            if target is None:
                continue
            criteria = questions[target]["criteria"]
            print(f"\n== tick {i} {target}, compaction {'on' if compact else 'off'}: {len(criteria)} options, first 6")
            for key, option in list(criteria.items())[:6]:
                print(f"  {key}: {json.dumps(option, ensure_ascii=False)[:110]}")
        for compact in (False, True):
            state, _ = shaped(tick["state"], tick["questions"], compact)
            rows = state["elements"][:4]
            print(
                f"\n== tick {i} state rows, compaction {'on' if compact else 'off'} (first 4 of {len(state['elements'])})"
            )
            for row in rows:
                print(f"  {json.dumps(row, ensure_ascii=False)[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
