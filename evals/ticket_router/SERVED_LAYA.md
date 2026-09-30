# Ticket router: in-process Laya against served Laya

Date: 2026-09-30. The plan below was fixed before the runs.

## Setup

| | how | Laya |
|---|---|---|
| C-in | `--model laya`, `LAYA_DEVICE=mps` | in process: fp32 weights, not compiled |
| C-direct | `--model laya-served`, `LAYA_SERVED_URL=http://127.0.0.1:8000` | system1-omni worker, `LAYA_WORKER_COMPILE=on LAYA_WORKER_WEIGHTS=fp16`, MPS |
| C-front | `--model laya-served`, `LAYA_SERVED_URL=http://127.0.0.1:8080` | the same worker process behind `omni-jev` |

- Checkpoint `convaiinnovations/laya` at `55cf4c4`, laya 0.3.20, torch 2.14.0.
- Hardware: M1 Pro (16 GB), macOS 26.1, on AC power.
- system1-omni at `9ad04e3`, the head of ThinkFlowLab/system1-omni#30 on 2026-09-30.
- system1-agents on branch `served-laya` at `0b3d23c`. The commits after it change only error wording, help text and this folder.
- Each configuration ran `s1a run ticket_router --model <m> --rethink off --seed 0 --episodes 3`. Seeds 0, 1 and 2 shuffle the same 30 tickets, giving 90 decisions per configuration.
- Order: the worker started once and was ready after 37 s. C-direct and C-front ran against it. The worker was then stopped and C-in ran, so no two models shared the GPU.
- The one-minute load average was 5.3–6.0 at the start of each configuration, from other work on the machine.

## Results

| config | correct | p50 ms | p95 ms | episodes s | model recorded | served on |
|---|---:|---:|---:|---:|---|---|
| C-in | 63/90 | 101 | 125 | 10.2 | `laya-rl-agent` | in process |
| C-direct | 63/90 | 77 | 94 | 7.8 | `convaiinnovations/laya@55cf4c4ebb4e` | mps, float16, compiled |
| C-front | 63/90 | 82 | 87 | 8.8 | `convaiinnovations/laya@55cf4c4ebb4e` | mps, float16, compiled, via `omni-jev` |

- Latency is per decision: the client round trip for the served configurations, the forward pass on a thread for C-in.
- Episode time is the sum over the three episodes. It leaves out process start and model load.

Routing agreement:
- All three configurations routed every one of the 90 (seed, ticket) pairs the same.
- C-in and C-direct differed by at most 0.001 in any probability.
- C-direct and C-front gave identical probabilities.
- The closest call in C-in had 0.0126 between its top two queues.

Both plan expectations held:
- C-direct and C-front agree everywhere.
- No ticket flips between the fp32 in-process model and the fp16 worker, so there are no differences to list.

C-in's lower speed comes from how Laya ran, fp32 and not compiled, against the worker's compiled fp16 model. It says nothing about HTTP cost. The frontend added about 5 ms at p50.

In-process runs record `laya-rl-agent`, the name laya reports. Served runs record the checkpoint and revision, plus `served_by` (device, dtypes, compile mode, time of the `/health` reading) in every tick.

## Reproduce

Start the worker and the frontend as in [docs/served-laya.md](../../docs/served-laya.md#8-run-it). Then, from this repository:

```sh
LAYA_SERVED_URL=http://127.0.0.1:8000 uv run s1a run ticket_router --model laya-served --rethink off --seed 0 --episodes 3
LAYA_SERVED_URL=http://127.0.0.1:8080 uv run s1a run ticket_router --model laya-served --rethink off --seed 0 --episodes 3
# stop the worker, then
LAYA_DEVICE=mps uv run s1a run ticket_router --model laya --rethink off --seed 0 --episodes 3
uv run python evals/ticket_router/compare_served.py C-in=<job dir> C-direct=<job dir> C-front=<job dir> \
  --json evals/ticket_router/served_laya_records.json
```

Job directories stay local, as for the other results here. [served_laya_records.json](served_laya_records.json) keeps every decision of the three runs: ticket, expected and predicted queue, probabilities, ms and the recorded model.
