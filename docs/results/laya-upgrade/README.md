# laya 0.3.5 → 0.3.20: same answers from `--model laya`

Reproduces the evidence for #37: `--model laya` gives the same answers on main and on that branch, on CPU and
on MPS, and the answers move only when Laya's MPS fp16 is turned back on.

| file | does |
|---|---|
| `run.sh` | checks out both commits, installs each one's lock, runs every case below, prints the tables |
| `laya_ab.py` | one fresh process: load through `LayaModel.from_env()`, answer the 30 ticket-router tickets, write a JSON |
| `compare.py` | the tables from those JSONs |

## Steps

1. Use a Mac with Apple silicon for the MPS cases (on other machines, step 4 with `DEVICES=cpu`), with
   `git` and `uv` installed and this repository cloned.
2. Fetch both checkpoints once, about 1.4 GB (`english` 804 MB, `multilingual` 614 MB):

   ```sh
   uv run --extra laya python -c "import laya; laya.load('convaiinnovations/laya', device='cpu'); laya.load('convaiinnovations/laya', device='cpu', subfolder='multilingual')"
   ```

3. Fetch the head commit. It is on #37's branch only, which a clone of ThinkFlowLab/system1-agents does not
   fetch (with `origin` pointing there):

   ```sh
   git fetch origin pull/37/head
   ```

4. From the repository root, with the baseline (main, laya 0.3.5) and head (#37's last code commit, laya 0.3.20):

   ```sh
   HF_HUB_OFFLINE=1 docs/results/laya-upgrade/run.sh . 3008e8f d0b3d29
   ```

   About 30 minutes on an M1 Pro. It writes to `evals/results/laya-upgrade` (ignored by git), or to the
   folder given as a fourth argument, which must be empty or new. `RUNS` (default 3), `DEVICES` (`cpu mps`)
   and `CHECKPOINTS` (`english multilingual`) narrow it. The checkouts go to a temporary folder and are removed
   at the end. Each run's output is in `logs/`; a failing run stops `run.sh` and prints the end of its log.
5. To compare with the recorded run, unpack `pr37-evidence.zip` from #37 and run
   `python docs/results/laya-upgrade/compare.py pr37-evidence/records`.

## What it runs

Each run is one process. It loads the model, then for each of the 30 tickets in
`s1a/agents/_data/ticket_router_eval.jsonl` asks one choice question, one noul question, and one request of
six questions (the choice and five nouls). Six questions is the case that changed: from 0.3.10 Laya answers a
request of five or more questions in fp16 on MPS, and #37 keeps it in fp32.

| case | runs |
|---|---|
| baseline and head, each checkpoint, each device | `RUNS` each, alternating baseline and head |
| head with `LAYA_MPS_AMP_MIN_ROWS=5` (Laya's default turned back on), MPS | 1 per checkpoint |
| baseline without its weight-init skip, CPU | 1 per checkpoint |

Every JSON records the commit, the laya, torch and transformers versions, the checkpoint revision, the
device and the Laya variables that were set.

## Reading the output

```
| checkpoint | device | load s, baseline | load s, head | ... | answers |
| convaiinnovations/laya | mps | 9.8 / 7.4 / 7.8 | 9.6 / 7.7 / 7.9 | ... | identical (3 + 3 runs) |
```

- `answers` is `identical` when every choice, probability, confidence and noul value of every run equals the
  first baseline run's. That is the claim; a `DIFFERENT` row disproves it.
- The `LAYA_MPS_AMP_MIN_ROWS=5` lines show what the fp32 default prevents: the largest probability change
  and each decision that changed, by ticket id. On an M1 Pro, `multilingual` changed two.
- The no-skip lines show the load the weight-init skip saves, now done by laya itself.
- Load and latency columns are for comparing the two sides of one row, run alternately on the same machine.
  The first load after the checkout can include reading the model files from disk.

## Recorded run

The 28 outputs of step 4 from 2026-10-05 are `records/` in `pr37-evidence.zip`, attached to #37: Apple M1 Pro (16 GB),
macOS 26.1, Python 3.14, torch 2.14.0, transformers 5.17.0, checkpoint revision `7b928d8`, baseline `222e656`
(laya 0.3.5), head `c197fe6` (laya 0.3.20). Those were main and #37's head before #37 was rebased onto main;
`c197fe6` is no longer on any branch. Against the commits in step 4, the path these runs exercise is unchanged: every
locked package has the same version, and `s1a/decision_models/laya.py` differs only by `check_window` moving to a
module function (#35) and the `bills_input_tokens` declaration (#17).
The machine had other work running (1-min load average about 6 to 10). All four rows are `identical`; with
`LAYA_MPS_AMP_MIN_ROWS=5`, `multilingual` changed `t_6cb98186dde0` (pick, logistics to returns) and
`t_f5251390fb88` (cancel, 0.4959 to 0.5074); without the skip the baseline loaded in 36.8 s (`english`) and
42.5 s (`multilingual`) against 6 to 10 s with it. The first `english` CPU load of each side (26.2 and 24.8 s)
read the model files from disk.

Not covered: the `typed-decisions` checkpoint, a local checkpoint path, transformers 4.x, CUDA, and Apple chips
other than the M1 Pro.
