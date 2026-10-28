# Screen a news desk, then wait for a person

## Summary

Run the five-stage desk in `s1a.content`: ingest, screen, human pick, fact-check, human publish decision.
The observable check is the fixture report: 50 labelled items screened, 4 picked items fact-checked, the
publish decisions equal to `publish.json`, and `human_publish.executed` empty. A finished JSON object with
`auto_publish: true` or a non-empty `executed` list is a failure even when the process exits 0.

The offline profile is a lexical rule, the same kind of baseline as `ticket_router`'s keywords. It matches
this fixture's labels by construction. That checks the wiring. It is not a quality claim for Jev, StartLux,
or PPLX. Deep research is the drafts already stored on the picked items. The run does not write them.

| Profile | Inference path | Requirements / validation |
|---|---|---|
| `--backend rule` (also `--backend auto` with `OMNI_BASE_URL` unset) | Lexical keep/drop and fact rule in-process | Core install, CPU, no model and no decision socket. Recorded below. |
| `--backend omni` | StartLux Instant 4B draft worker, `POST /v1/systemone`, short prefill | `OMNI_BASE_URL`. Draft [system1-omni #164](https://github.com/InferMatrixLab/system1-omni/pull/164) does not need to be merged. Command not run here. |
| `--backend omni --fact-check-worker pplx` | Screen stays StartLux; fact-check uses the PPLX draft worker | `OMNI_FACTCHECK_BASE_URL`. Draft [system1-omni #166](https://github.com/InferMatrixLab/system1-omni/pull/166). Command not run here. |
| Clef | Refused | [system1-omni #165](https://github.com/InferMatrixLab/system1-omni/pull/165) is not on this hot path. |

## Prerequisites and setup

From a System1-Agents checkout, Python 3.11+ and `uv`:

```bash
uv sync --frozen
uv run python -m s1a.content run --help
mkdir -p runs/recipes/content-workflow
git rev-parse HEAD
```

Keep the printed revision with the result. The core install is enough. No browser, GPU, model download, or
API key is required for the rule profile. `MODEL_NAME` is unused: this path does not call the chat model.

The labelled sample is `items.jsonl` (50 items, 32 `keep`, 18 `drop`, 8 drafts). `picks.json` names four of
the kept drafts. `publish.json` records the human decision, including `publish` for an item the fact-check
calls unsupported. `sample.rss` and `x_posts.jsonl` are the offline RSS and X adapters. URLs in the sample
use `example.invalid` and are not fetched.

File sha256 of `items.jsonl` at the recorded run: `51828b6671f507e4ce635d90a43420b1855fde864c43dfc17c072e3ba983f064`.

## Run the task

```bash
OMNI_BASE_URL= OMNI_API_KEY= uv run python -m s1a.content run \
  --items recipes/content-workflow/items.jsonl \
  --picks recipes/content-workflow/picks.json \
  --publish recipes/content-workflow/publish.json \
  --backend rule \
  > runs/recipes/content-workflow/report.json
```

Stdout is one JSON object. A missing gate prints one line on stderr and exits 2. `--backend rule` does not
read `OMNI_BASE_URL`. The shell prefix clears it so a developer environment cannot switch this command onto
a worker.

## Verify the result

The check uses the fixture labels, the pick file, and the publish file. It fails on a short screen, a wrong
choice or score, a skipped gate, or anything that looks like a post:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("runs/recipes/content-workflow/report.json").read_text())
items = [json.loads(line) for line in Path("recipes/content-workflow/items.jsonl").read_text().splitlines() if line.strip()]
picks = json.loads(Path("recipes/content-workflow/picks.json").read_text())["ids"]
publish = json.loads(Path("recipes/content-workflow/publish.json").read_text())["decisions"]
labels = {row["id"]: row["label"] for row in items}
fact_labels = {row["id"]: row["draft"]["label"] for row in items if row.get("draft")}
screen = {row["id"]: row for row in report["screen"]["records"]}
assert report["backend"] == "rule" and report["unsafe_skip"] is False
assert report["stages"] == ["ingest", "screen", "human_pick", "fact_check", "human_publish"]
assert report["screen"]["total"] == report["screen"]["labeled"] == 50
assert report["screen"]["kept"] == 32
assert report["screen"]["choice_accuracy"] == report["screen"]["score_accuracy"] == 1.0
assert report["ingest"]["sources"]["rss"] > 0 and report["ingest"]["sources"]["x"] > 0
assert sum(report["ingest"]["sources"].values()) == 50
for ident, label in labels.items():
    assert screen[ident]["choice"] == label
    assert screen[ident]["choice_correct"] and screen[ident]["score_correct"]
assert report["human_pick"]["ids"] == picks
facts = report["fact_check"]["records"]
assert [row["id"] for row in facts] == picks
for row in facts:
    assert row["choice"] == fact_labels[row["id"]]
    assert row["choice_correct"] and row["score_correct"]
    print(row["id"], "->", row["choice"], "score", row["score_level"], "decision", publish[row["id"]])
assert report["human_publish"]["decisions"] == publish
assert "publish" in publish.values()
assert report["human_publish"]["executed"] == [] and report["auto_publish"] is False
assert report["latency_gate"]["attached"] is False
print("screen", report["screen"]["choice_correct"], "/", report["screen"]["total"])
print("Verified the fixture workflow; nothing was published")
PY
```

The expected closing line is `Verified the fixture workflow; nothing was published`. Exit 0 without that
line is not success. A stopped gate keeps a partial packet only when `--packet` is set, and that packet is
not this report.

Recorded rule-profile result, parent revision `9b20b989d919bc69227f81bc0ec571ca56925586`, before the commit
that added this recipe: screen 50/50, fact-check `n01 supported`, `n02 unsupported`, `n03 needs_human`,
`n04 supported`, publish decisions unchanged, `executed` empty. The same command without `--picks` exited 2
with `human pick gate` on stderr. Details: [smoke-evidence.json](smoke-evidence.json).

## Use a draft Omni worker

Do not point this path at Clef. StartLux is the default screen and fact-check worker. PPLX is an optional
second server for fact-check only. Neither draft pull request has to be merged; the process only needs the
URL. No live worker was running for the recorded run, so the blocks below are unchecked commands.

Readiness, then one screened run. `OMNI_API_KEY` is sent as a bearer token only when it is non-empty.

```bash
curl --fail "$OMNI_BASE_URL/health"
uv run python -m s1a.content run \
  --items recipes/content-workflow/items.jsonl \
  --picks recipes/content-workflow/picks.json \
  --publish recipes/content-workflow/publish.json \
  --backend omni \
  > runs/recipes/content-workflow/report.json
```

The report's `workers.screen` names the model, the draft pull request, and `prefill: short`. Each screen
record's `calls` has two request ids, `screen` and `relevance`. Agreement with the lexical labels is a
comparison, not a pass bar: a worker that disagrees is still a successful desk run if the gates held and
nothing was published. Re-run the verifier's structural checks, and do not require `choice_accuracy == 1`
for a worker. Keep the worker's misses in the report.

Heavier fact-check on a second URL:

```bash
uv run python -m s1a.content latency-report --report runs/recipes/content-workflow/report.json
# after three measured rounds, warmup excluded, and only if you have a ceiling:
# uv run python -m s1a.content latency-report --report report.json --p95-max-ms <ceiling> --min-samples <n>
```

```bash
OMNI_FACTCHECK_BASE_URL=http://127.0.0.1:8001 uv run python -m s1a.content run \
  --items recipes/content-workflow/items.jsonl \
  --picks recipes/content-workflow/picks.json \
  --publish recipes/content-workflow/publish.json \
  --backend omni --fact-check-worker pplx \
  > runs/recipes/content-workflow/report.json
```

`latency-report` with a ceiling and no Omni samples fails closed (exit 1). That is the hook for a later
Public231 C1 gate: warmup 5 excluded, concurrency 1, three rounds, then p50/p95. The clock is
`client_request_wall_ms`, not Omni's `post_to_body_json_and_schema_validation`. No ceiling is pinned. See
[the workflow doc](../../docs/content-workflow.md#latency-gate-not-attached).

One item can also be asked as a choice through `s1a decide`. That command has no score head, needs a Jev
key, and was not run for this recipe:

```bash
uv run s1a decide \
  --state '{"title": "Open model checkpoint lands for local agents", "summary": "A small checkpoint for agent inference is up for local tests.", "source": "rss"}' \
  --option keep="Relevant AI, model, agent, inference, chip, checkpoint or research hotspot" \
  --option drop="Not a relevant hotspot for this desk" \
  --rules "Keep a relevant hotspot for an AI-agent news desk. Drop sports, weather, recipes, traffic, celebrity and anything without that hook."
```

## Demo and validation

The required application + System1-Agents + System1-Omni video is not in this recipe. No draft worker was
served here, and pull requests 164 and 166 are unmerged. A rule-backend terminal run does not fill that
gap. The recipe stays an evidence gap for that video until a maintainer accepts the exception or a later
run supplies the clip: input items, the worker's choice and score, the human files, `executed: []`, and
the answering model, request id, and base URL from the same run.

`tests/test_content_workflow.py` covers the fixture, both gates, the unsafe flag, RSS and X fixtures, and
a mock StartLux/PPLX transport. It does not open a socket except the CLI test that expects a refused
connection at `127.0.0.1:9`.

## Troubleshooting and limits

| Symptom | Check |
|---|---|
| Exit 2, `human pick gate` | Pass `--picks`, answer the prompt on a terminal, or read the unsafe flag's help before using it |
| Exit 2, `human publish gate` | The stderr line lists fact-check choices. Pass `--publish`. Re-running repeats the screen |
| Exit 2, pick was not kept | The id was dropped. Pick from `screen.kept_ids` |
| `PPLX fact-check needs OMNI_FACTCHECK_BASE_URL` | PPLX is a different draft server. Do not send it to the StartLux port by default |
| `Clef is not on the content hot path` | Use StartLux, or PPLX for fact-check only |
| Omni HTTP error | The message has the status, not the body. Check `/health` and the model alias |
| `choice_accuracy` below 1 on `omni` | Expected until a worker is measured. The rule profile is the wiring check |

The lexical rule is brittle: the word "model" in "model trains" would be kept. This fixture does not
contain that sentence. Fifty synthetic items are not a desk accuracy claim. RSS URLs and X are not
contacted by the recorded command. A human `publish` value is not a post.

## References

- [Workflow doc](../../docs/content-workflow.md) and [configuration](../../docs/configuration.md)
- [Ticket-routing recipe](../ticket-routing/README.md), the batch-decision analogue
- [Decision-model interface](../../docs/decision-models.md)
- Draft workers: [StartLux #164](https://github.com/InferMatrixLab/system1-omni/pull/164), [PPLX #166](https://github.com/InferMatrixLab/system1-omni/pull/166). Clef [#165](https://github.com/InferMatrixLab/system1-omni/pull/165) is excluded.
