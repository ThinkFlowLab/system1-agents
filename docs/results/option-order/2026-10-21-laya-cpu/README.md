# Option-order check: in-process LAYA on CPU, ticket-router probe cases (2026-10-21)

First real run of `s1a probe --order` (#90). One backend, one case file, one machine; a robustness measurement,
not an accuracy claim and not a rate for other checkpoints or hardware.

- Agents: this PR's branch `issue-90-option-order-flip-rate`, based on `main` `5693f51`.
- Model: `--model laya`, in-process `laya` 0.3.20, checkpoint `convaiinnovations/laya@7b928d828b7b0e022f929d9bd2e44165aa270148`
  (English; laya warned that some shipped temperatures were clamped, so its own confidence is uncalibrated).
- Device: CPU only (torch 2.14.0, no CUDA device visible), Linux x86_64, 8 vCPU Intel Xeon, 15 GB RAM.
- Cases: `evals/ticket_router/probe.jsonl`, 12 cases, 5 options each.
- Date: 2026-10-21 (Asia/Shanghai).

| Command | Flips | Accuracy, file order | Accuracy, reordered asks | Median ms |
|---|---|---|---|---|
| `uv run s1a probe evals/ticket_router/probe.jsonl --model laya --order reverse --json laya-reverse.json` | 3/12 (25%) | 9/12 (75%) | 6/12 (50%) | 420 |
| `uv run s1a probe evals/ticket_router/probe.jsonl --model laya --order shuffle:5 --seed 0 --json laya-shuffle5.json` | 3/12 (25%) | 9/12 (75%) | 39/60 (65%) | 247 |

Under `reverse` the flips were "negated background" (returns → logistics), "background payment" (account → payment)
and "negation insufficient" (human → logistics); two of the three moved from a right answer to a wrong one. Under
five shuffles, "insufficient information" flipped instead of "negation insufficient". The JSON files keep every
order asked (`rows[].asks[].options`) and the key chosen in it. Latency is CPU wall time per decision on a shared
box and is not a performance claim.

Not covered here: the saved-run replay over recorded ticket-router/browser requests (scope item 2 of #90), other
backends (Jev, CLM, served Laya), CUDA/MPS, and the multilingual checkpoint.
