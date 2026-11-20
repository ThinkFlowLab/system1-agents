# Cross-question interference on `decide_many`

[Delaunay (2026)](https://arxiv.org/abs/2611.12521) reports that bundling several System 1 questions in one request is not the same as asking them one at a time: later heads can attend to earlier ones. This is a different failure mode from option-order sensitivity (`s1a probe --order`, #90) and from confidence semantics (#89). `s1a probe --bundle` measures it on this repository's `decide_many` path. There is no mitigation here.

## What is bundled in this repository

Browser ticks are the in-repo production bundled path. `docs/architecture.md` and `s1a/browser/decision_model.py` send every head of the action space in one `decide_many` (typically `operation`, then one `{op}_target` per operation, and `text_value` when values are offered). Ticket-router today sends only `{"pick": ChoiceQuestion(...)}`, so it is a single-question path. The committed file uses it as a **negative control** by adding a dummy second question next to a ticket-style Choice, not as the main suite.

## How to run it

The cases live in `evals/probe/interference.jsonl`: six ticket-like Choice+Noul items on one state, one ticket-router negative control (a dummy Noul, "the ticket record includes an id field"), and one modest browser-shaped frozen tick (`operation`, `click_target`, `type_text_target` in production head order). The browser tick is a constructed frozen observation and question set, not a replay of a recorded episode: committed `decision_ticks.json` files do not keep the original questions or the observation.

```bash
uv run s1a probe evals/probe/interference.jsonl --model laya --bundle --json docs/results/decide-many-interference/laya-bundle.json
```

Hosted Jev is the same command with `--model jev` when a key is already in the environment. `--bundle` and `--order` cannot be used together. Without `--bundle`, the fit probe and `decide_many` are unchanged.

For each case the probe:

1. Asks every head in its own request (isolated).
2. Asks the same questions in one `decide_many`, in file / production order (bundled).
3. Asks them again with the question-name order reversed (directional interference).
4. Repeats the bundled request in the same order once (the non-determinism control, the same idea as `--order same:N`).

It reports, per backend and per head, the key or binary flip rate versus the isolated answer (a Noul flips when `p >= 0.5` changes) and the mean total variation between the probability distributions. Both answers stay in the record and in the `--json` file.

## Results

No live numbers are committed yet. A first `laya` CPU row needs `uv sync --extra laya` and the `convaiinnovations/laya` checkpoint; a `jev` row needs a TypeSafe or OpenRouter key. Do not copy the paper's Tev / Clef percentages as Agents results. When a run is recorded, put the command, Agents SHA, checkpoint, device, date and the JSON next to [docs/results/decide-many-interference/](results/decide-many-interference/README.md).
