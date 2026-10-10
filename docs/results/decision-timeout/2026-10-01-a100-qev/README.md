# The #36 description's OneJev-27B runs, every one kept (2026-10-01 and 2026-10-02, one A100 80 GB)

The records behind the table in #36's description ("OneJev-27B, 5 s: 0 / 5; 30 s: 7 / 7"), including the three 30 s
runs that were interrupted by hand and left out of the count, plus two later OneJev-27B batches that are not in that
table. Real server, no injected delay.

## Setup

- Model and server: OneJev-27B (`OmniJev/OneJev-27B`) through `qev serve --multimodal` on port 8000; s1a reaches it
  with `TYPESAFE_API_URL=http://127.0.0.1:8000/v1/systemone TYPESAFE_MODEL=jev-latest` and `--model jev`.
  This is not system1-omni and not Open-Jev; see `../2026-10-08-a40-native/` for the system1-omni runs.
- Task: `s1a run flights --timeout 600 --profile-out <run>.json`, one-way Zurich to London on November 1, 2026.
- Hardware: one NVIDIA A100 80 GB PCIe. Success is judged by Gemini 2.5 Flash on the final screenshot
  (`runs/*.png`), as in the `bench*.csv` files.
- **How the 30 s deadline was set:** these runs predate #36's code. A test-only launcher, `slow_ok.py`, replaced
  `JevModel.from_env`'s default deadline with 30 s; `S1A_DECISION_TIMEOUT_S=30` does the same since #36.
  The 5 s runs used the default deadline. `compact_ok.py` is the same launcher plus a #9-style compact page state.
  `bench.sh` is the batch script as it is now, after these runs.

## The table in #36, run by run

| run (file prefix) | deadline | decisions | outcome | counted in #36 |
|---|---|---|---|---|
| `onejev27B_125333` to `onejev27B_125659` (5 runs started 2026-10-01 12:53 to 12:56) | 5 s (default) | 7 each | BLOCKED at decision 8, the date on the calendar page: `decisions connection failed` | yes, 0 / 5 |
| `onejev27B_135818`, `onejev27B_135952` | 30 s | 11 each | flights shown | yes |
| `onejev27B_140127` | 30 s | none recorded | **interrupted by hand** (Ctrl+C, `KeyboardInterrupt` in the log) | no |
| `onejev27B_140231`, `onejev27B_140246` | 30 s | 0 | **interrupted**: the first request failed 3 to 8 s after start (`decisions connection failed`) while the batch was being stopped by hand, as #36's description says | no |
| `onejev27B_151748` to `onejev27B_152407` (5 runs) | 30 s | 11 each | flights shown | yes |

30 s: ten runs started, seven completed and counted (7 / 7), three interrupted and not counted. Over the seven:
model time per task 60.3 to 62.8 s (median 61.2 s), median decision 3.2 to 3.8 s (median of medians 3.7 s), page
11.3 to 11.4 s, browser 9.6 to 9.9 s (`bench_30s.csv`). #36's description gives 9.7 s for the browser median; the
seven values give 9.9 s.

Where the 5 s deadline fails, decision by decision (run `onejev27B_151748`, 30 s): decisions 1 to 7 take 1.0 to
3.8 s; decision 8, picking November 1 on the calendar page (15,062 input tokens against 5,745 before it), takes
13.2 s; decision 9 (Done) 17.3 s. With 5 s, decision 8 is cut, which is where every 5 s run stops.

## Not in the #36 table

| run (file prefix) | deadline | outcome |
|---|---|---|
| `onejev27Bc_152539` to `onejev27Bc_153103` (5 runs started 2026-10-01 15:25 to 15:31) | 30 s, compact page state (`compact_ok.py`) | 4 flights shown; 1 Google error page ("Oops, something went wrong") |
| `onejev27B_094702` to `onejev27B_095250` (5 runs started 2026-10-02 09:47 to 09:52) | 30 s, after installing flash-linear-attention on the server | 5 flights shown; median decision 2.6 to 2.7 s (`bench_30s_fla.csv`) |

## Files

- `runs/<prefix>.log`: each run's stdout and stderr (the final JSON with status, answer and actions, or the
  traceback). `runs/<prefix>.json`: the `--profile-out` record with every decision's time, tokens and choice (absent
  for the 5 s runs, which were not profiled, and for `140127`). `runs/<prefix>.png`: the final screenshot.
- `bench_5s.csv`, `bench_30s.csv`, `bench_30s_fla.csv`: the batch rows (time is the run's end), with the judge's
  verdict.
- `slow_ok.py`, `compact_ok.py`, `bench.sh`: the launchers and batch script. Home paths are shortened to `~/`.

Privacy and reuse: the browser was signed out; screenshots are Google Flights pages, kept as evaluation evidence.
