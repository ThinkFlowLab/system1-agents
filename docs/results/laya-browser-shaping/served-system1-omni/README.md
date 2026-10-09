# Served Laya through system1-omni, with this PR's shaping

The same 12 recorded Flights ticks (`../zurich-london.jsonl`), shaped by `laya_state()` and `laya_browser_question()`
exactly as `LayaModel._decide` does (`make_bodies.py` writes `requests.jsonl`), sent three ways with byte-identical
state and questions (`served_trial.py`):

| arm | path | window asked |
|---|---|---|
| `served-1536` | system1-omni's Rust frontend `/v1/systemone` → its Laya text worker (`recipe/laya`, `laya-serve`) | `max_len=1536`, `head_max_len=1024` in the request body |
| `served-default` | the same, no window fields | the worker's checkpoint window |
| `inproc-1536` | `laya.load()` in the worker's own environment, `system_one(..., max_len=1536, head_max_len=1024)` | 1536 / 1024 |

## Result: the served path cannot take the browser window yet

- The path works end to end: 12 ticks, 45 questions, no error.
- **`served-1536` answered exactly as `served-default` on all 45 questions, with the same token counts.** The window
  fields never reached the model: system1-omni's recipe pins `laya[serve]==0.3.20`, whose `laya-serve` reads no
  `max_len` or `head_max_len` from the body; the Rust frontend forwards the body unchanged. So the worker runs the
  checkpoint's window, 512 tokens per question.
- On the calendar tick (8), the in-process call reads 3,629 tokens over its 3 questions; the served one 1,536, i.e.
  512 per question. Served and in-process agree on 36 of 45 answers; the 9 differences are 8 `click_target` heads and
  the calendar's operation (`DONE` served, `BLOCKED` in process), the heads the larger window changes.
- Tick 9 shows 4,096 served tokens over 4 questions, more than 512 each: consistent with the worker's router sending
  that tick to the multilingual checkpoint (1,024 window). The response's model field was not recorded, so this is
  not confirmed.
- Neither arm drives the task: both answer `DONE` from tick 7 on, and `DONE` at tick 0, as in the live run.

## What a served trial equivalent to the in-process one needs

1. A newer `laya[serve]` in system1-omni's Laya recipe: `laya-serve` 0.3.28 reads `max_len` and `head_max_len` from
   the request body (capped by `LAYA_MAX_TOKEN_BUDGET`, 8,192 by default).
2. A served client that sends that window and applies this PR's shaping. The served adapter of #35 builds its body
   with `laya_question()` only: the browser compaction in `LayaModel._decide` does not run on that path, and it takes
   the window from the checkpoint (`LAYA_SERVED_MAX_LEN`, 512 by default).

## Setup

- system1-omni `ae86032cba2466f45f42c2ebdfadbcfa8c30eb9f` (2026-10-06), `omni-jev` release build, Laya text worker
  on CPU (`LAYA_DEVICE=cpu LAYA_MODELS=english LAYA_THREADS=4`), `laya 0.3.20`, checkpoint `convaiinnovations/laya`.
- Agents: this branch at 9f87fed for the shaping.
- Linux x86_64, 8 CPU cores, no GPU used. Median per tick: 6.3 s served, 7.2 s in process (CPU).
- `run_served.sh` is the command; `run_served.log` its output; `served_results.jsonl` one line per tick and question,
  with Jev's recorded answer next to each arm.
