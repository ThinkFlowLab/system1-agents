# Decision model benchmarks

Compare an existing decision service on JevBench's 231 public questions: easy48,
original72 and hard111. The collector saves every request and response; a separate
command re-scores them with the pinned upstream scorer.

[H800 results](results/public231-h800.md) compare five open checkpoints. This is the
public231 set, not the article's 248-question or 23-model leaderboard.

## Run

Use the repository's development environment and a running service. Clone
[JevBench](https://github.com/fstandhartinger/jevbench) at
`7ce310c7262ed49cc85853339a8a42459298e3f3`; the tool verifies the content hashes of
all three datasets and every scorer module it imports. No dataset is vendored.

```bash
export JEVBENCH_ROOT=/absolute/path/to/jevbench
python -m benchmarks.legs.leg1_public231 predict \
  --endpoint http://127.0.0.1:8000 --path /v1/systemone \
  --model open-jev-9b --thinking off --warmup 5 --rounds 3 \
  --run-meta /absolute/path/to/environment.json \
  --out /absolute/path/to/results/open-jev-9b
python -m benchmarks.legs.leg1_public231 aggregate \
  --run /absolute/path/to/results/open-jev-9b \
  --out /absolute/path/to/results/open-jev-9b
```

Omni native Open-Jev uses `/v1/systemone`; Intern's official service uses
`/v1/decisions`. `--model` is optional. The collector preserves state, questions and
option order, sends `thinking.enabled=false`, and never sends targets or provenance.
It makes one request at a time, without retries. It does not start services or choose GPUs.

`environment.json` describes the **actual service**, without credentials:

```json
{
  "checkpoint": {"repo": "owner/model", "revision": "full-checkpoint-commit"},
  "backend": {"name": "serving-backend", "version": "full-source-commit"},
  "sampling": {"temperature": 1.0},
  "hardware": {"gpu": "GPU model"}
}
```

Also record dtype, tokenizer/base/adapter/head revisions, loaded libraries, context
limits, cache settings and job identity. The file is copied byte-for-byte and hashed;
all original fields must match the run metadata. Omit `stop_reason` and conflicting
collector fields. Temperature is recorded, not configured by the client. Authenticate
outside this tool if needed; it does not accept or save API keys.

## Reading the results

- Each round's accuracy is **correct / 231**. Failures and unattempted questions stay
  in the denominator. Three repeated rounds provide timing samples, not 693 unique questions.
- Upstream schema validity, Brier and ECE are reported separately. Missing probabilities
  are not replaced with one-hot predictions. Native `noul` scalars become `{no: 1-p, yes: p}`.
- Per-round and pooled p50/p95 cover HTTP POST through full response receipt. All attempts,
  successes and failures have separate counts; empty latency sets are null. Warmups are excluded.
- `sufficient=true` requires five warmups and three complete 231-question rounds.
  `--limit`, fewer rounds or missing attempts produce partial observations.
- HTTP 401/403/429, interruption or three consecutive infrastructure failures stop collection.
  HTTP 422 counts as a failed question and does not trigger that stop counter.

Evidence goes in a fresh directory **outside the repository**. Each round contains
`results.jsonl` and `raw/<sha256(task_id)>.json`; warmups are separate. Raw records retain
the request, response text, decoded response, HTTP status, error, timestamp and latency.
`run.meta.json` and `completion.json` record the plan and any stop reason.

Aggregation checks source hashes, the environment, input order, raw hashes and record
consistency before re-scoring. Orphan or missing raw files are errors. It never overwrites
an existing summary; use a new output directory for another recomputation. These checks
detect inconsistent evidence, not coordinated rewriting of an unsigned archive.

## Tests

```bash
JEVBENCH_ROOT=/absolute/path/to/jevbench python -m pytest tests/benchmarks -q --confcutdir=tests/benchmarks
ruff check benchmarks tests/benchmarks
ruff format --check benchmarks tests/benchmarks
```

Reference-dependent tests skip if `JEVBENCH_ROOT` is absent. Mock and loopback tests
check collection and scoring behavior; model quality and latency come from the linked real runs.
