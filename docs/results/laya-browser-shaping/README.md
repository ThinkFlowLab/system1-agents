# Laya browser input shaping: evidence

What Laya receives on the browser front with `LAYA_COMPACT_BROWSER_STATE` off and on, on the same recorded input,
and the one live Google Flights run. Laya's own tokenizer and `build_sequence` produce every count; no model forward
pass is needed for them.

## Files

| file | what it is |
|---|---|
| `zurich-london.jsonl` | 12 browser ticks recorded from a Jev 1.13 run of the `flights` task (Zurich to London, one adult, economy), the `state` and `questions` exactly as the browser front sent them, and Jev's `answers`. File dated 2026-09-29; predates 0fbfc7a. The account button's label is the generic "Google Account"; no personal data. Page labels and text are Google Flights' own, kept for evaluation only; the `????` are characters the recorder could not encode. |
| `trace.txt` | the output of the command below: per tick and per question, state tokens (and how many fit the window), option tokens, and whether the options were cut to an equal share, compaction off and on, under two windows; then tick 8 (the calendar, 66 elements) as Laya reads it both ways. |
| `live-run-2026-09-28/` | `answer.json` and `decision_ticks.json` of the live run reported in the PR. |
| `regression-tests.txt` | the regression tests that pin the current shaping, the probe's string flags and the WAIT record. |

## Reproduce

```bash
uv sync --extra laya
uv run python scripts/laya_shaping_trace.py docs/results/laya-browser-shaping/zurich-london.jsonl --show 8
uv run pytest -v tests/test_decision_models_laya.py -k "TestLayaState or TestLayaBrowserQuestion"
uv run pytest -v "tests/test_browser_policy.py::TestJevWaitCollapsesIntoInPageSettling::test_a_wait_that_moved_the_page_is_recorded_as_progress"
```

Revisions: shaping code at 0fbfc7a plus the commit that adds this folder; tokenizer `convaiinnovations/laya` at
revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Environment of the trace: Windows 11, CPU, Python 3.13.14,
laya 0.3.5, transformers 5.17.0, torch 2.14.0+cpu.

## What the trace shows

Windows: `512 / 192` is the checkpoint's default (`max_len` / `head_max_len`); `1536 / 1024` is what the PR
recommends for browser runs (`LAYA_MAX_LEN` / `LAYA_HEAD_MAX_LEN`).

| | compaction off | compaction on |
|---|---|---|
| state tokens per tick | 256 to 4,726 | 91 to 998 |
| `click_target` option tokens (25-element page) | 640 to 672 | 136 to 152 |
| `click_target` option tokens (calendar, 66 to 67 elements) | 3,225 to 3,275 | 695 to 717 |
| `512 / 192`: options cut to an equal share | every `click_target` head of 10 elements or more (4 to 19 tokens per option) | the three `click_target` heads of 55 to 67 elements (4 tokens per option) |
| `512 / 192`: state kept | at most 316 tokens; the whole state only on the 3-element tick | the whole state on ticks 0 to 5; 232 to 439 tokens on ticks 6 to 11 |
| `1536 / 1024`: options cut | the calendar and results `click_target` heads (15 to 20 tokens per option) | none |
| `1536 / 1024`: state kept | 508 to 922 tokens of the 1,173- to 4,726-token states | the whole state on every question except the two calendar ticks' `click_target` (775 of 926, 753 of 998) |

So with the shaping and the browser window, every option reaches Laya whole on every recorded tick, and the state
fits except on the calendar's target head, where the 66 day buttons take the room. Under the default window the
shaping is not enough for those heads.

The calendar's target options (tick 8), first lines of `trace.txt`:

```text
off: 3: {"element": "[3] Friday, September 25, 2026 ????", "current_value": "", "role": "button", "region": "dialog:Ba...
on:  3: "Friday, September 25, 2026 ?"
```

Under `512 / 192` without the shaping, each of these options gets 4 tokens, so Laya reads the start of the JSON
(`{"element": "[`) and no date.

## The live run (2026-09-28, before ecefeea and 0fbfc7a)

```bash
LAYA_MAX_LEN=1536 LAYA_HEAD_MAX_LEN=1024 uv run s1a run flights --model laya --timeout 900
```

Code: the tree committed as e513234 nine minutes later (state and question shaping, before the review fixes);
checkpoint `convaiinnovations/laya` at `55cf4c4`; Windows 11, CPU only, Chrome over CDP.

`decision_ticks.json`: one tick, 25 elements, 1,254 input tokens over four questions, 12,397 ms. Operation
probabilities: DONE 0.548, CLICK 0.225, BLOCKED 0.087, WAIT 0.058, SCROLL_DOWN 0.049, TYPE_TEXT 0.033.

`answer.json`: `status` DONE with `steps: 0`, `interactions: 0` and an empty `history`; the final page is the
Google Flights home page (`https://www.google.com/travel/flights?hl=en`) with nothing filled, and the chat model's
answer says no Zurich to London flight is shown. The run's DONE is Laya's verdict at the first step, not a
completed task.

## Live runs with video (2026-10-06, at 70227de)

```bash
LAYA_MAX_LEN=1536 LAYA_HEAD_MAX_LEN=1024 PLAYWRIGHT_MCP_COMMAND=python PLAYWRIGHT_MCP_ARGS="-m evals.replay.cast --frames <dir>/frames -- node <@playwright/mcp@0.0.78>/cli.js --cdp-endpoint=http://127.0.0.1:9224"   python -m s1a run flights --model laya --timeout 600 --logs-dir <dir>
```

Code at 70227de (this PR with every review fix); checkpoint `convaiinnovations/laya` at `55cf4c4`, in process;
laya 0.3.5, Python 3.13.14; Windows 11, CPU only; one frame after every browser call (`evals/replay/cast.py`).
Each folder holds `decision_ticks.json`, `answer.json` and `calls.jsonl` (every browser call on the frame clock).
The videos are those frames at their own timestamps, real time, no cuts, under a one-line title band.

| folder | browser | what Laya did | video |
|---|---|---|---|
| `live-run-2026-10-06-annotated/` | the same Chrome and profile, at 69ad3e5, run through `dump_laya.py`, which appends every Laya call's exact shaped state, questions and answer to `laya_calls.jsonl` | DONE at the first decision (0.435; DONE 0.687, CLICK 0.179) on the home page, 1,209 input tokens over 3 questions, nothing filled: a failed task | 41 s, attached to the PR: a slide with the shaped input, the frames in real time with the decision under them, a slide with the outcome (`annotate_video.py`) |
| `live-run-2026-10-06/` | Chrome 153 on its own profile over CDP, signed out, Google's consent refused once before the run | DONE at the first decision (confidence 0.294, 1,583 input tokens) on the Google Flights home page, nothing filled: a failed task | 30 s, attached to the PR |
| `live-run-2026-10-06-consent-page/` | `@playwright/mcp`'s own isolated, fresh profile, headed | started on Google's cookie consent page; clicked "Sign in", then "Create account" four times, DONE (0.043) on the sign-in page after 64 s: a failed task | 65 s, attached to the PR |
