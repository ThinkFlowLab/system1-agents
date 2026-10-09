# `S1A_DECISION_TIMEOUT_S` on system1-omni's native Open-Jev worker (2026-10-08, one NVIDIA A40)

The same three runs as [`../2026-10-07-a40/`](../2026-10-07-a40/), now through system1-omni's own native
Rust/CUDA Open-Jev worker, which serves Open-Jev-9B since system1-omni 4a79980. Real-server runs, no injected delay.

## Serving path

s1a `flights --model jev` (main at 22e685b, the #36 merge), `TYPESAFE_API_URL=http://127.0.0.1:8080/v1/systemone`,
`TYPESAFE_MODEL=jev-latest` → system1-omni's Rust frontend `omni-jev` → `omni-open-jev-native` at 4a79980 (CUDA
library built for sm_86), `OPEN_JEV_MODEL=weights/open-jev-9b-merged`: `ZefanCai/Open-Jev-9B@47e9668` on
`Qwen/Qwen3.5-9B@c202236`, exported with `recipe/open_jev/export_merged.py` on CPU (15 s, peak 18.5 GiB RAM, 15 GiB
output). The worker held 15.7 GB of the A40 and answered the recipe's example request (7 candidates) in 0.26 s.
`native9b.sh` is the setup as run. The `Open-Jev 3308a15` line and the empty second health line in `versions.txt`
are left over from the reference-server run script and do not apply here.

Open-Jev-27B-v1.1 still does not fit this GPU (see `../2026-10-07-a40/openjev-27b-native/worker.log`).

## Runs

| folder | deadline | decisions | end | what happened |
|---|---|---|---|---|
| `default-5s/` | 5 s (unset) | 0 | BLOCKED at 12.3 s | the first decision was not answered within 5 s (the same first decision takes 8.7 s in the 30 s run); `[181001] model call failed, reason: decisions connection failed`; no hang |
| `timeout-30s/` | `S1A_DECISION_TIMEOUT_S=30` | 2 | finished at 23 s | both decisions answered, 8.7 s (SCROLL_DOWN) and 6.8 s (BLOCKED, confidence 0.40); the model's BLOCKED ended the run |
| `timeout-1s/` | `S1A_DECISION_TIMEOUT_S=1` | 0 | BLOCKED at 5.4 s | a deadline the server always exceeds: clean failure about 1 s after the first decision is asked, no hang |

The task is not solved in any of them: with 30 s every decision completes, and the 9B model itself answers BLOCKED at
its second decision on the Google Flights home page. A decision takes 7 to 9 s here because this worker scores every
candidate with its own prompt and runs the 9B's candidates one at a time (the recipe validates candidate packing on
27B only).

No run was interrupted or left out.

## Files

`<arm>/answer.json`, `<arm>/decision_ticks.json`, `<arm>/calls.jsonl`, `versions.txt` (commits, health, GPU), and
the scripts as run (`native9b.sh`, `record_timeout.sh`, `annotate_timeout.py` for the video attached to #36).
Same hardware and browser as `../2026-10-07-a40/`. Privacy and reuse: as there.
