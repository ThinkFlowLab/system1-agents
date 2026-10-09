# Route support tickets to five queues

## Summary

Route a local batch of support requests to `logistics`, `payment`, `returns`, `account` or `human` with the
existing `ticket_router` agent. The input is a ticket's ID, title and description; the agent enumerates five
choices and executes one route per ticket. Expected labels are evaluator-only and are omitted from its state.

This recipe uses five synthetic tickets, one per queue. Completion means all five were processed without an
execution error; the verification below also checks that their predicted routes match the fixture's labels.
It exercises the local agent loop and produces inspectable episode artifacts. The fixture is deliberately
simple and does not measure classification quality on real support traffic.

| Profile | Inference path | Requirements / validation |
|---|---|---|
| `--model rule` | Deterministic keyword baseline (`keywords` in the run record) | Core install, CPU, no model or API calls; see recorded smoke evidence below |
| `--model laya` | Laya in process | `laya` extra and checkpoint; CPU selected below; command checked, real-model run not recorded for this recipe |
| `--model jev` | TypeSafe Jev over HTTP | TypeSafe or OpenRouter decision key; command checked, live endpoint run not recorded for this recipe |
| `--model laya-served` | System1-Omni Laya worker through `omni-jev` | Prepared worker/frontend; this fixture's serving run and video remain pending |

## Prerequisites and setup

Use a System1-Agents checkout, Python 3.11+ and `uv`. Run all commands from its root:

```bash
uv sync --frozen
uv run s1a run ticket_router --help
mkdir -p runs/recipes/ticket-routing
git rev-parse HEAD
```

Keep the printed source revision with your result. The locked core install supplies the openJiuwen runtime;
the first sync can fetch dependencies. See [CONTRIBUTING.md](../../CONTRIBUTING.md#setup) for installation details.
The baseline requires no browser, GPU, model download or API credential. An empty exported `MODEL_NAME`
disables the optional chat fallback, including a value configured in `.env`.

## Run the task

Start with the rule baseline:

```bash
MODEL_NAME= uv run s1a run ticket_router --model rule --rethink off \
  --dataset recipes/ticket-routing/tickets.jsonl --batch-size 5 \
  --episodes 1 --seed 0 --max-steps 5 --timeout 60 --showcase --log \
  > runs/recipes/ticket-routing/summary.json
```

`--log` prints the chosen routes to stderr after the episode. Stdout is one JSON summary containing `job_dir`.
`--showcase` puts the job under `evals/showcase/`, outside the benchmark matrix. If `S1A_HOME` is set, that job
uses its output root; the shell redirection above still writes under the current checkout.

## Verify the result

This check uses the fixture's labels and the actual episode routes. It fails on incomplete processing,
execution errors, invalid keys or a wrong route:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("runs/recipes/ticket-routing/summary.json").read_text())
episodes = list(Path(summary["job_dir"]).glob("*/agent/episode.json"))
assert len(episodes) == 1, "expected one recorded episode"
episode = json.loads(episodes[0].read_text())
report = episode["extra"]["ticket_router"]
expected = {
    row["id"]: row["label"]
    for row in map(json.loads, Path("recipes/ticket-routing/tickets.jsonl").read_text().splitlines())
}
assert summary["errors"] == 0 and summary["invalid_keys"] == 0, summary
assert report["processed"] == report["total"] == len(expected), report
assert not report["unprocessed_ids"], report
actual = {row["id"]: row["predicted"] for row in report["routes"]}
assert actual == expected, actual
for row in report["routes"]:
    print(row["id"], "->", row["predicted"], "expected", row["expected"])
print("Verified all five fixture routes")
PY
```

The expected final line is `Verified all five fixture routes`. The order of rows follows the seeded shuffle.
Retain `summary.json`, the trial's `agent/episode.json` and `result.json`; the episode includes inputs/views,
decisions, routes, coverage and the batch fingerprint. An exit code of zero alone is insufficient: a stopped
episode can retain a partial score and unprocessed tickets.

## Use a decision model

After the baseline works, install the optional Laya backend. Its first load may download a checkpoint; prepare
that separately from a recording. This profile selects CPU explicitly:

```bash
uv sync --frozen --extra laya
MODEL_NAME= LAYA_DEVICE=cpu uv run s1a run ticket_router --model laya --rethink off \
  --dataset recipes/ticket-routing/tickets.jsonl --batch-size 5 \
  --episodes 1 --seed 0 --max-steps 5 --timeout 60 --showcase --log \
  > runs/recipes/ticket-routing/summary.json
```

Run the same verifier and preserve its actual outcome. Record the checkpoint/subfolder, resolved checkpoint
revision, Laya version and window settings. A failed route remains a failed result; this recipe has no recorded
Laya quality result. Save each profile's summary before another command replaces it.

For hosted Jev, configure `TYPESAFE_API_KEY` for the direct service or `OPENROUTER_API_KEY` for the proxy using
[the configuration guide](../../docs/configuration.md), then use the baseline command with `--model jev`.
`MODEL_NAME=` still disables chat fallback; Jev decision calls are external and billed.

### Required System1-Omni demo profile

[System1-Omni supports Laya workers](https://github.com/ThinkFlowLab/system1-omni/blob/main/docs/supported-models.md),
and the client from merged [PR #35](https://github.com/ThinkFlowLab/system1-agents/pull/35) registers
`--model laya-served`. Follow [the served setup](../../docs/served-laya.md#8-run-it) for the worker and Rust
frontend, then record its ready endpoint and this fixture's agent run:

```bash
curl --fail http://127.0.0.1:8080/health
MODEL_NAME= LAYA_SERVED_URL=http://127.0.0.1:8080 uv run s1a run ticket_router --model laya-served --rethink off \
  --dataset recipes/ticket-routing/tickets.jsonl --batch-size 5 \
  --episodes 1 --seed 0 --max-steps 5 --timeout 60 --showcase --log \
  > runs/recipes/ticket-routing/summary.json
```

Run the same verifier and show the input tickets, actual selected routes and checked outcome. Include the
answering `model`, `served_by`, frontend `url` and `request_id` from this run's ticks, with both repositories'
commits and worker/checkpoint/device settings. Preserve a failed route as a failed outcome.
Use [PR #35's video and caption](https://github.com/ThinkFlowLab/system1-agents/pull/35#issuecomment-6017425034)
as the example. It covers three tickets from a different dataset and does not validate this five-ticket fixture.

## Demo and validation

Record the terminal command, ticket inputs, actual selected routes and the verifier's outcome using the
[video guide](../../CONTRIBUTING.md#agent-video-demos). Important PRs require the application/task +
System1-Agents + System1-Omni video from the served profile above. The rule and in-process runs provide
additional evidence and do not fulfill that requirement. Show failures or interventions
and link the run artifacts. The historical ticket-router replay in the root README covers a different workload.

The recipe's smoke validation and source/environment details are recorded in [smoke-evidence.json](smoke-evidence.json).
No video, live Jev run, Laya inference or System1-Omni fixture run is recorded for this recipe yet.
Its full demonstration remains an explicit evidence gap until that run and video are supplied.

## Troubleshooting and limits

| Symptom | Check |
|---|---|
| Missing package / checkpoint | Install the matching extra and prepare its model before the run; retain load errors |
| Jev key/configuration error | Check the decision-service variables; they are separate from chat-model settings |
| Unprocessed tickets | Inspect episode errors and step/time budgets; processing five tickets needs five accepted actions |
| Wrong route | Retain the route/verifier failure; a valid choice is not necessarily the correct classification |
| Unknown `laya-served` model | Check the agent revision and client registration; worker support alone does not add the client |

The bundled labels are evaluation data. The agent supports neither ticket-system writes nor production-service
authentication here; adapt its environment separately for an actual helpdesk. The keyword baseline ignores
nuance such as negation and quoted background; five easy samples provide no general accuracy claim.

## References

- [Agent implementation](../../s1a/agents/ticket_router.py) and [agent guide](../../docs/agents.md#the-ticket-router)
- [Shared loop and evaluation protocol](../../evals/README.md)
- [Decision-model interface](../../docs/decision-models.md)
- [Recipe authoring template](../TEMPLATE.md)
