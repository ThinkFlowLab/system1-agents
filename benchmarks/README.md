# Decision model benchmarks

Compare an existing decision service on JevBench's 231 public questions: easy48,
original72 and hard111. The collector saves every request and response; a separate
command re-scores them with the pinned upstream scorer.

[H800 results](results/public231-h800.md) compare five open checkpoints. This is the
public231 set, not the article's 248-question or 23-model leaderboard.

The [accuracy leaderboard](accuracy-leaderboard.md) ranks that frozen run and
defines how to add comparable results. Request latency has a separate
[System1-Omni leaderboard](https://github.com/ThinkFlowLab/system1-omni/pull/131).

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

## Offline quality from shared raw

This consumer re-scores saved Omni requests without another model run.
The existing `predict` and `aggregate` commands and their
default timing basis are unchanged.

For public231, provide the frozen input manifest, the independently frozen Omni
`bench.py` interface (`5aa20a2c…`), and one `--run` path per planned round in order:

```bash
export JEVBENCH_ROOT=/absolute/path/to/pinned/jevbench
python -m benchmarks.quality_public231 \
  --input-manifest /absolute/path/to/input-r1/manifest.json \
  --collector /absolute/path/to/omni-raw-interface-v2/bench.py \
  --model-config-id open-9b \
  --run /absolute/path/to/round-1 --run /absolute/path/to/round-2 \
  --run /absolute/path/to/round-3 \
  --out /absolute/path/to/new-quality-output
```

Each raw directory must contain the complete frozen `requests.jsonl`, `config.json`,
`completion.json` and saved `responses.jsonl` when measurement started. The Omni
interface checks file/response hashes, IDs, decoding and its saved wire verdict.
The quality consumer checks the pinned inputs and request order, then uses the
original JevBench scorer on the raw HTTP/typed probabilities. An Omni
`invalid_response` can still be a valid relaxed JevBench distribution. Wire validity,
distribution validity, strict validity, renormalization and label correctness remain
separate. Choice/score hard labels are never substituted for missing probabilities;
noul scalars use `{no: 1-p, yes: p}`. Accuracy uses the original argmax rule, including
lexical tie breaking; ordinal expected value is used for MAE.

Missing round directories and started requests without a saved terminal remain
in the plan. Three rounds have 231 unique tasks and 693 planned observations.
The ordered CLI paths define the round plan; the producer saves one directory per
round. A saved `metadata.round`, when present, must match that ordered round;
the remaining model/configuration metadata must agree between available rounds.
The resulting `summary.json` and full-plan `records.jsonl` carry original raw
pointers and hashes. Quality output uses null timestamps/latencies; it does not
report speed. The raw clock is POST through body decoding, JSON and schema validation,
and the saved response is decoded HTTPX text, not compressed wire bytes.

Use a fresh output directory outside the repository. The tool checks archive
consistency; it does not authenticate a coordinated rewrite of unsigned evidence.
CPU fixtures exercise the reader and scorer, not model quality or GPU execution.

The required benchmark CI job fetches the pinned JevBench reference and Omni
collector, generates the public231 fixture, and rejects skipped tests. To run the
same suite locally, set `JEVBENCH_ROOT` and `OMNI_BENCH_SOURCE` to those checkouts:

```bash
python -m pytest tests/benchmarks -q --confcutdir=tests/benchmarks
```

The reader accepts the original collector (`5aa20a2c…`). Each archive must match
the exact collector supplied with `--collector`; the probability rules, input
bytes and timing basis are unchanged.
