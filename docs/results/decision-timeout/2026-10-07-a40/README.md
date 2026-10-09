# `S1A_DECISION_TIMEOUT_S` on a real server (2026-10-07, one NVIDIA A40)

Evidence for #36: the same Flights task on the same server, three runs one after the other, only the per-decision
deadline changing. Real-server runs, no injected delay. The unit tests in `tests/test_decision_models_jev.py` are the
separate, injected-value evidence.

## Serving path

s1a `flights --model jev`, `TYPESAFE_API_URL=http://127.0.0.1:8080/v1/systemone`, `TYPESAFE_MODEL=jev-latest`
→ system1-omni's Rust frontend `omni-jev` (`OMNI_JEV_BIND=127.0.0.1:8080`) → Open-Jev's reference server
(`python -m jev.server`, port 8791) with **Open-Jev-9B** (`ZefanCai/Open-Jev-9B@47e9668` on
`Qwen/Qwen3.5-9B@c202236`, bf16, batch size 32, prefix cache off).

This is not system1-omni's native Open-Jev worker, and not the OneJev-27B / `qev serve` setup of the PR description:

- **Open-Jev-27B-v1.1 through the native worker** (`omni-open-jev-native`, built for sm_86) does not fit one A40.
  Its merged export is 48 GiB and the worker loads every text weight onto CUDA device 0; on the 46 GB A40 it stops
  at load with `Error: cudaMalloc: out of memory (2)` (`openjev-27b-native/worker.log`). It needs a single GPU with
  more memory (an A100 80 GB, H100 or H200).
- The export itself works on this machine: `recipe/open_jev/export_merged.py` on CPU took 37 s with a peak resident
  memory of 59 GiB (the recipe says roughly 110 GB), output 48 GiB (`openjev-27b-native/export.txt`).
- Open-Jev-9B is a different checkpoint from both Open-Jev-27B-v1.1 and OneJev-27B. It is the Open-Jev model that
  fits this GPU, served by its own reference server behind system1-omni's frontend.

## Runs

| folder | deadline | decisions | end | what happened |
|---|---|---|---|---|
| `default-5s/` | 5 s (unset) | 0 | BLOCKED at 13.5 s | the first decision was not answered within 5 s; `[181001] model call failed, reason: decisions connection failed`; no hang |
| `timeout-30s/` | `S1A_DECISION_TIMEOUT_S=30` | 7 | finished at 104 s | every decision answered, 7.1 to 18.0 s each (median 13.5 s), none cut; see the outcome below |
| `timeout-1s/` | `S1A_DECISION_TIMEOUT_S=1` | 0 | BLOCKED at 7.3 s | a deadline the server always exceeds: the first decision fails cleanly about 1 s after it was asked, no hang |

Outcome of the 30 s run: the agent scrolled down and up, typed Zurich and London, pressed Enter three times, and the chat model then answered
"€102" from the round-trip calendar (Nov 8, 7-day trips) without a one-way search. The deadline let the run go on; the
task itself is not solved. With 5 s the same server cannot answer the first decision (13.5 s here), so the run never
starts acting.

No run was interrupted or left out: these are the only three runs made with this setup.

## Files

- `<arm>/answer.json`, `<arm>/decision_ticks.json` (every decision: operation, target, confidence, `decision_ms`),
  `<arm>/calls.jsonl` (every browser call and its time, from `evals/replay/cast.py`).
- `versions.txt`: exact commits (system1-agents 22e685b, the #36 merge; system1-omni 99865743; Open-Jev 3308a15),
  the server health answers, the GPU and driver.
- Scripts as run: `build_omni.sh` (system1-omni CUDA library for sm_86, worker and frontend), `export_openjev.sh`,
  `serve_openjev9b.sh`, `record_timeout.sh` (the three runs), `annotate_timeout.py` (the video attached to #36).

Hardware: one NVIDIA A40 46 GB (driver 615.71.09), 48 vCPU, 377 GB RAM, Ubuntu 24.04, Linux 6.8. Browser: Playwright's
Chromium build 1194 headless, 1280x900, its own profile, signed out, Google's consent refused once.

## Privacy and reuse

The browser was signed out and the records hold no account or personal data. Screenshots are Google Flights pages,
kept as evaluation evidence. The video may be reused in project updates with a link to this folder.
