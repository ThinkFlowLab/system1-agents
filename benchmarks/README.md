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

The public231 collector uses the typed `state` / `questions` / `thinking` request
and reads `answers.decision`. Omni native text Open-Jev exposes this contract at
`/v1/systemone`; Intern's official service uses `/v1/decisions`. Omni's request
and response schema depends on the selected worker: native JEV-27B-VL instead
uses the flat contract described below at the same `/v1/systemone` path.
`--model` is optional. The collector preserves state, questions and option order,
sends `thinking.enabled=false`, and never sends targets or provenance.
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

## Changed-image replay

Replay a frozen screenshot sequence using the flat `kind` / `state` / `question` /
`options` contract of official JEV `/v1/decide` or Omni native JEV-27B-VL
`/v1/systemone`, using the same `DecisionClient` without a model override. This
collector does not send the public231 typed contract to Intern `/v1/decisions`:

```bash
python -m benchmarks.legs.changed_image_replay \
  --manifest /absolute/path/to/sequence.jsonl \
  --endpoint http://127.0.0.1:8000 --path /v1/systemone \
  --run-meta /absolute/path/to/environment.json \
  --cache-state unknown --out /absolute/path/to/results/image-replay
```

Each JSONL line has a unique nonempty `id`, a nonempty `group`, and the original
`request` object. For example (replace the placeholder with actual base64 bytes):

```json
{"id":"frame-1","group":"game-over","request":{"kind":"choice","state":["Inspect the screenshot.",{"image":"data:image/png;base64,..."}],"question":"Game over?","options":["No","Yes"]}}
```

`state` is a list with at least one inline PNG/JPEG image, either `{"image": URI}`
or `{"type":"image_url","image_url":{"url": URI}}`. Preflight checks every row,
base64 encoding and MIME signature before any request; image decoding stays with
the service. Remote URLs are rejected. `choice` requires 2–256 string options;
`noul` and `score` omit options. The original request, image order and option order
are preserved. Repeated requests with distinct IDs are sent again. There are no
warmups, retries, deduplication, service restarts or cache resets.

Run metadata requires `checkpoint.repo/revision` and `backend.name/version` from
the actual service. Include source revisions, hardware and cache configuration
as available. `--cache-state` records only the caller's declaration: the caller
must arrange a cold service, and resetting counters does not empty its caches.
Keep any canonical-ID alias map alongside the external input evidence.

The fresh output directory must be outside this repository. It retains the exact
`manifest.jsonl`, `environment.json`, `run.meta.json`, one `raw/NNNNNN.json` per
attempt and `results.jsonl`. Result records include sequence/id/group/kind,
`request_sha256` (of the request serialized by `benchmarks.evidence.dump`), ordered
`image_uri_sha256`, `response_body_sha256`, and the raw file's path and hash.
The full request, image URIs and original response text are in raw evidence,
together with HTTP status, decoded response, client error, timestamp and latency.
`completion.json` counts planned/attempted/remaining/succeeded/failed and records
the stop reason.
HTTP 400/422 refusals and malformed answers remain failed records; access errors,
other non-2xx responses, transport failures and interruptions stop the sequence.

Response validation checks kind/effective_kind, ordered options, probability
count/range, a probability sum within `1e-6` of 1, and choice/index consistency.
The sum tolerance only allows floating-point rounding, including float32 output;
it is not a reference-comparison tolerance. Unnormalized answers remain
`parse_error` records with their raw evidence and count as failures. `noul` uses
`false,true`; `score` uses `0` through `5`. `ok` and `succeeded` mean only a valid
response; they do not establish reference agreement, cache correctness or task
completion, and do not add three-kind support to the Agents `DecisionModel` API.
Use the independent Omni harness for numerical reference comparison. Comparing
repeated images requires the complete request to match, including all state text,
image order, question, options and inference settings. Compare the cache path
with the reference or cache-disabled path under the same checkpoint and runtime;
require deterministic inference settings or account for expected nondeterminism
before attributing differences to caching. Matching image URI hashes alone is
insufficient. Client `latency_s` ends after `response.text` is read, before JSON
decoding; it excludes service startup. Short replays do not establish stable p95
latency, and replaying 2048 screenshots does not establish autonomous game completion.

## Tests

```bash
JEVBENCH_ROOT=/absolute/path/to/jevbench python -m pytest tests/benchmarks -q --confcutdir=tests/benchmarks
ruff check benchmarks tests/benchmarks
ruff format --check benchmarks tests/benchmarks
```

Reference-dependent tests skip if `JEVBENCH_ROOT` is absent. Mock tests
check collection and scoring behavior; model quality and latency come from the linked real runs.
