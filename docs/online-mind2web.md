# Online-Mind2Web live slice

A seeded 16-task **live-web** slice of [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web)
(Xue et al., 2025, [arXiv:2504.01382](https://arxiv.org/abs/2504.01382); dataset
[osunlp/Online-Mind2Web](https://huggingface.co/datasets/osunlp/Online-Mind2Web), CC BY 4.0). It answers "does the
full agent finish a realistic live-web task?", which is a different question from:

- the offline step-level collector in issue #109 / PR #111 (frozen page, right operation/element)
- the Allrecipes / WebVoyager task 0 row already in [benchmarks.md](benchmarks.md)
- the planned Anthropic `browser_toolset` baseline in issue #104

## The slice

Ids, levels, start URLs and the official revision live in `evals/online_mind2web/slice.json`. Task text is **not**
vendored. The official file is gated: accept the dataset terms, set `HF_TOKEN`, then download the pinned revision
(the 2026-05-15 task updates are in this SHA):

```bash
# HF_TOKEN must already be able to read osunlp/Online-Mind2Web
uv run python -m evals.online_mind2web fetch
uv run python -m evals.online_mind2web select --official .external/online_mind2web/Online_Mind2Web.json
```

`select` re-derives the 16 ids from the official file and seed `116`. The rule is: take the official tasks whose
text does not ask for a login, payment, cart, favorites or similar account write; `random.Random(116).sample` 12
sorted easy ids and 4 sorted medium ids; commit that list sorted by id. At this revision the public-read pool is 76
easy and 135 medium tasks.

## Running

Each attempt starts a new `browse` session in its own `logs_dir` (no shared browser profile) and uses the official
start URL plus the official task text as the goal. The first backend is hosted Jev (`--model jev`); pass `--arms jev
llm` to add the chat-model arm under the same `--max-steps` and `--timeout`.

```bash
uv run python -m evals.online_mind2web run --official .external/online_mind2web/Online_Mind2Web.json --arms jev
```

`results.md` and `config.json` name the Agents SHA, the dataset revision, the model, the budget and the date. Every
planned attempt is in the denominator, including timeouts, agent errors, `BLOCKED` and unevaluated episodes.

## Judging

Success is [WebJudge](https://github.com/OSU-NLP-Group/Online-Mind2Web), the official auto-eval, at the upstream
default threshold of 3 (upstream now recommends GPT-6 Luna as the backbone). This repository does not vendor
WebJudge. Without a judge, a finished episode is recorded as `unevaluated`, never as a pass. After an external
WebJudge run you can apply integer scores with `--judge-scores scores.json` (a JSON object keyed by `task_id` or `task_id__arm`; the unsuffixed `task_id` is the fallback).

## Results

No live numbers are committed yet: a run needs `HF_TOKEN`, a Jev (and optionally chat-model) key, and a live network.
Place traces under `docs/results/online-mind2web/` and redact secrets.

## Licence

The dataset is CC BY 4.0; the WebJudge / evaluation code in the official repository is MIT. Attribution is in
`NOTICE` and `THIRD_PARTY_LICENSES.md`. Only ids, websites, levels and the revision are committed here.
