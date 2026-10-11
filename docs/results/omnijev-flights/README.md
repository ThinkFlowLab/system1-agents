# OmniJev-4B on Google Flights: every s1a run

Every `s1a run flights --model omnijev` run on the A100 server that wrote a record, and the matched Jev runs. These
are s1a runs; the four OmniJev GIFs in the README are OmniJev's own recorded-trajectory replays, not these.

Task: `flights`, one-way Zurich to London on November 1, 2026, one adult, economy; stop when matching flights show.

## What each run did

| run | code | decisions | end | final page | outcome |
|---|---|---|---|---|---|
| `2026-09-29__14-18-26` | 1d01e69 | 25 | form budget | form | form filled, then Enter in Departure instead of Search; wandered (Frankfurt card, main menu) |
| `2026-09-29__14-22-54` | 1d01e69 | 14 | BLOCKED | results | every step, Search included; Google answered "No results returned. Oops, something went wrong." |
| `2026-09-29__14-54-07` | 1d01e69 | 25 | form budget | form | form filled, lost after it |
| `2026-09-29__14-56-33` | 1d01e69 | 25 | form budget | form | form filled, lost after it |
| `2026-09-29__14-58-50` | 1d01e69 | 25 | form budget | form | form filled, lost after it |
| `2026-09-29__15-01-10` | 1d01e69 | 17 | BLOCKED | form | lost, blocked |
| `2026-09-29__15-02-15` | 1d01e69 | 4 | BLOCKED | form | WAIT with no progress after 3 actions |
| `2026-09-29__15-03-06` | 1d01e69 | 7 | BLOCKED | form | started 2 s before the next one on the same Chrome: not a valid run |
| `2026-09-29__15-03-08` | 1d01e69 | 7 | BLOCKED | form | the other half of that pair, not valid either; both typed into "Return" and had no value for it |
| `2026-09-29__15-04-13` | 1d01e69 | 25 | form budget | form | form filled, lost after it |
| **`2026-09-29__15-06-35`** | 1d01e69 | 15 | **DONE** | results | **every step, Search included; answer "easyJet, €72, 9:30 PM to 10:20 PM, November 1", the flight Jev found** |
| `2026-09-29__15-08-19` | 1d01e69 | 25 | form budget | form | form filled, lost after it |
| `2026-09-30__09-37-51` to `09-47-20` (5 runs) | e2bf4c5 | 25 each | form budget | form | form filled, then lost (location menu, deal pages) |
| `2026-09-30__15-36-07`, `15-38-08`, `15-44-39` | e2bf4c5 | 25 each | form budget | **results** | form filled, scrolled, clicked Search at decision 23; the flights show, but the budget ended before DONE |
| `2026-09-30__15-40-07` | e2bf4c5 | 25 | form budget | form | opened "Flights from popular destinations" and started over |
| `2026-09-30__15-42-13` | e2bf4c5 | 25 | form budget | form | stuck on the location pop-up |
| `2026-09-30__15-32-35` to `15-35-27` (5 runs) | e2bf4c5 | 12 to 13 | DONE | results | **Jev 1.13**, the matched runs: same server, Chrome and profile |

Counted: 20 OmniJev runs (22 records, the concurrent pair left out). 1 completed (DONE on the right flight), 4 more
reached the results page (one Google error, three without DONE before the budget), 15 did not. Before the e2bf4c5
history fix (1d01e69, 2026-09-29): 1 of 10 completed. After it (2026-09-30): 0 of 10 completed, 3 reached the results. Jev: 5 of
5 completed, 12 to 13 decisions, 283 to 367 ms median per decision.

The PR's first results table listed seven of the 2026-09-29 runs (14-18-26, 14-22-54, 15-02-15, 15-03-0x, 15-04-13,
15-06-35, 15-08-19). The other four that day (14-54-07, 14-56-33, 14-58-50, 15-01-10) were not in it; they are here.
Four earlier folders that day (14-13-24, 14-17-35, 14-46-20, 14-51-28) wrote no record and are not counted.

## The vision judge's verdicts (2026-09-30 afternoon)

`2026-09-30/benchmark/` holds that batch's final screenshots, logs and `bench.csv` with Gemini 2.5 Flash's verdicts
on each screenshot. It marked all five OmniJev runs as failed. For `15-36-07`, `15-38-08` and `15-44-39` the
screenshot does show Zurich to London flights on November 1 (the results URL's search date is 2026-11-01); the judge
read the train suggestion and the "Nov 1" chip as a wrong match. Those three are counted above as "reached the
results, no DONE", not as completions: the task asks the agent to stop there, and it did not.

## Files

| file | what it is |
|---|---|
| `2026-09-29/<run>/`, `2026-09-30/<run>/` | each run's `answer.json` (status, final page, the chat model's answer) and `decision_ticks.json` (every decision: operation, target, confidence, decision and screenshot time) |
| `actions.txt` | every run's final page and actions, from `summarize.py` over those records |
| `2026-09-29/logs/` | the terminal output of the 15:02 to 15:08 batch (`run1.log` to `run5.log`) and the launcher `run_flights_gpu.sh` |
| `2026-09-30/benchmark/` | the afternoon batch: final screenshots (`bench_<model>_<time>.png`, one per run, no replay, no cuts), logs, `bench.csv`, and the scripts (`bench.sh`, `final_shot.py`, `judge.py`; `bench.sh` as it is now, its `jev` and `omnijev` lines unchanged since that run; the two Python scripts reformatted by ruff, otherwise as run) |

No screenshots exist for the 2026-09-29 runs and the 2026-09-30 morning runs: their evidence is the records above.

## Revisions and setup

- system1-agents: 1d01e69 on 2026-09-29 (before the e2bf4c5 history fix), e2bf4c5 on 2026-09-30 (pulled at 09:37 UTC,
  first run 09:37:51).
- OmniJev: clone at 14dbec4 (`OMNIJEV_REPO`). Checkpoint `tinnel123/OmniJev` release v1.1, the 4B (LoRA rank 32 on
  `Qwen/Qwen3.5-4B`), local copies as `OMNIJEV_CHECKPOINT` and `OMNIJEV_BASE`. Prompt mode `full` (the default;
  `OMNIJEV_PROMPT` unset).
- Hardware: one NVIDIA A100 80 GB PCIe, Linux. Without `flash-linear-attention` and `causal_conv1d` (the logs say
  so). On the 2026-09-30 afternoon the GPU also held CLM's Qwen3-8B server (half the memory); one allocation
  warning shows in `bench_omnijev_153604.log` and the run went on.
- Browser: Playwright's Chromium build 1194, headless (`--headless=new --no-sandbox`, 1280x900), its own profile that
  nothing else used, Google's cookie banner refused once before the runs, `@playwright/mcp` 0.0.78 over CDP.
- Budgets: `s1a run flights --model omnijev --timeout 600`; openJiuwen's browser subagent ends a run on its form-phase
  budget (`form_phase_budget_exhausted`, 25 decisions here) and its 240 s browser budget.
- Jev: `s1a run flights --model jev --timeout 600`, Jev 1.13 through OpenRouter, same server and profile, e2bf4c5.

## Privacy and reuse

The browser was signed out ("Sign in" shows on every screenshot); no account or personal data is in the records.
Screenshots and page text are Google Flights' pages, kept as evaluation evidence only.

## Recorded runs on CPU (2026-10-06, at 6d9c015)

OmniJev-0.8B v1.1 in process on Windows 11 CPU (no GPU), OmniJev clone at 14dbec4, prompt `full`, one frame after
every browser call (`evals/replay/cast.py`), `s1a run flights --model omnijev --timeout 1200`.

| folder | browser | what happened |
|---|---|---|
| `2026-10-06-cpu/live-run-0.8b/` | Chrome 153 on its own profile over CDP, signed out, Google's consent refused once | two decisions, 136 s and 128 s, both CLICK "Change ticket type. Round trip"; openJiuwen's task deadline ended the run at 301 s (`task_deadline_exhausted`) |
| `2026-10-06-cpu/live-run-0.8b-consent-page/` | `@playwright/mcp`'s own isolated, fresh profile | started on Google's consent page; CLICK "Language: English" (55 s), the page went blank, WAIT, BLOCKED "waited without progress" |

The video attached to the PR is the first run's frames at 4x speed, no cuts (44 s). At CPU speed the 240 s browser
budget allows two or three decisions; the GPU runs above are the task-level evidence.

## Recorded runs on GPU (2026-10-07, at 0829ce0)

OmniJev-4B v1.1 in process on one NVIDIA A40 46 GB (driver 615.71.09, Linux 6.8), OmniJev clone at 14dbec4, prompt
`full`, Playwright's Chromium build 1194 headless on its own profile, Google's consent refused once, one frame after
every browser call; three runs in a row, all kept (`record_omnijev4b.sh`; `versions.txt` has the exact commits).
The task date is `GOAL_DATE`, here November 8, 2026.

| folder | decisions | median decision | end | what happened |
|---|---|---|---|---|
| `2026-10-07-a40/run1/` | 25 | 1.8 s | form budget | steps 1 to 9 (47 s) fill the form right: Zurich, London, one way, Sun Nov 8, Done; then Enter in Departure instead of Search, scrolls, the location pop-up, the main menu, "Flights from New York" |
| `2026-10-07-a40/run2/` | 25 | 2.0 s | form budget | the same 9 steps, then scrolls up and down without clicking Search |
| `2026-10-07-a40/run3/` | 25 | 1.8 s | form budget | the same 25 decisions as run 1 |

0 of 3 completed; none reached the results page. This is the most common outcome class of the 20 A100 runs above
("form filled, lost after it"). The video attached to the PR is run 1 in real time, no cuts, with each decision
under its frame (`annotate_omnijev4b.py`). Each folder holds `answer.json`, `decision_ticks.json` and `calls.jsonl`
(the browser calls and their times).

## Serving

system1-omni has no OmniJev vision worker: its supported models are Laya (text), Open-Jev-27B (text) and Cua-S1
(text and screenshot). `--model omnijev` therefore runs OmniJev in process only; a served path would need an OmniJev
worker in system1-omni first.
