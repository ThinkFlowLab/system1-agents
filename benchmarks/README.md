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

## Offline quality from shared raw

These consumers re-score saved Omni requests and original ticket episodes without
another model run. The existing `predict` and `aggregate` commands and their
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

Ticket quality needs the existing System1-Agents environment, including openJiuwen.
Public231 does not need that framework. Use the frozen ticket producer r3 preparation
and the **whole** measurement root, including unsuccessful or missing cells:

```bash
python -m benchmarks.quality_ticket \
  --preparation /absolute/path/to/ticket-gpu-prepare/r3 \
  --run /absolute/path/to/measurement-root \
  --out /absolute/path/to/new-ticket-quality-output
```

This version accepts the frozen `open-9b`, `open-27b` and `laya-english` cells,
seed0, one episode, batch30, maxsteps30, timeout300 and decision deadline120.
It reuses the original fixture/environment and choice validator, checks every
POST/retry against the current public ticket and queues, and follows original
validated slot → accepted act → committed environment step → returned job.
Decisions and actions must occur inside the original `Runner.run_agent` call;
cleanup/release, episode return, annotate/close, write_job and stop must follow
the original caller's order. Laya-served response provenance is checked with the
original model's response augmentation or captured health/routing fallback;
other wire fields remain exact comparisons.
Labels come only from the fixed fixture. Requests, probabilities/confidence,
before/after observations, original job ticks/views, loaded source and cell/asset
identity are checked; archived report scores are diagnostic comparisons. Non-decision
HTTP bodies are also checked for exact saved bytes, hash and viewing text. Captured
worker/frontend executables must match launch paths and recorded asset hashes, and
the ready/after process IDs and maps must remain consistent.
Remote artifact paths are relocated only inside that cell's recorded arm prefix.

Each cell reports fixture-derived `route_correct / 30`, `accepted_processed / 30`,
`complete_episode`, and `strict_batch_success` (normal complete episode and all
30 routes correct). Timeout/error/missing evidence retain verified prefixes and
the full 30 denominator. `MODEL_NAME` must be empty, rethink off, and original
episode `rethink` must be boolean `false`, with `chat_calls` zero. Route counts
describe individually verified calls under the recorded cell identity. Damaged
cleanup, process/asset cross-checks or framework copies block complete/strict
results without removing those closed routes; a broken ticket association blocks
that ticket. `process_identity_state` reports that separate process/asset check.
`answer` is an explicitly **provisional** normal-type gate;
missing, empty or incomplete framework source captures cannot be complete/strict.
all results require independent review of the actual captured framework source.
Other/unknown result types cannot be complete or strict successes. Exact recorded
asset/source consistency does not establish historical binary build ancestry.

Both tools require fresh output outside the repository. They detect inconsistent
unsigned archives, not coordinated rewriting of all evidence. No real model quality,
GPU execution or framework normal-type acceptance is established by their CPU mocks.

The new tests additionally use `OMNI_BENCH_SOURCE`, `PUBLIC231_INPUT_MANIFEST`, and
`TICKET_PREPARATION_ROOT` pointing to those frozen external inputs. When the CPU
environment lacks openJiuwen, ticket tests stub only framework imports and execute
the repository's actual environment/validator; that is separate from real framework
integration. Run the benchmark tests with the same pytest command above.
