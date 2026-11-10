# Content workflow

Jev screens. A person decides. The desk path is five stages:

1. Ingest news from RSS and, when an adapter is supplied, Twitter/X.
2. Screen each item to keep or drop, and score relevance.
3. A person picks 3 to 5 kept items for deep research.
4. Fact-check those items' claims against the sources the person supplied, as a choice and a score.
5. A person decides publish or hold. The run records that decision and posts nothing.

System1-Agents owns ingest, the gates, and the report. A draft System1-Omni worker is used only for stage 2 and stage 4, and only when `OMNI_BASE_URL` is set. The shared tool loop is not used: that loop would act on every choice, and these two gates must be able to stop the run.

The diagram that asks for about 200 items down to 30–50, then 3–5, is the shape of a live desk. The shipped fixture is 50 labelled items and keeps 32. It does not measure the diagram's 15 seconds or ¥0.01.

## Where each stage lives

| Stage | Code | What it refuses to do |
|---|---|---|
| Ingest RSS | `load_rss_file` / `load_rss_url` in `s1a/content/ingest.py` | A path is a file. A URL is opt-in via `--rss-url`. |
| Ingest X | `load_x_fixture`, `posts_to_items`, `UnconfiguredXAdapter` | No request is sent to `api.x.com`. `X_BEARER_TOKEN` is not read. |
| Screen | `RuleDecider.screen` or `OmniDecider.screen` | Labels and drafts are not copied into the prefill. |
| Human pick | `run_workflow` in `s1a/content/workflow.py` | Missing picks raise `GateRequired` before fact-check. |
| Fact-check | `RuleDecider.fact_check` or `OmniDecider.fact_check` | No draft means `needs_human` and no model call. The worker does not write the research. |
| Human publish | the same `run_workflow` | `human_publish.executed` is always `[]`. `auto_publish` is always false. |

A live X adapter is a `fetch()` callable passed to `load_inputs` as `x_fetch`. It should `GET https://api.x.com/2/tweets/search/recent` with `Authorization: Bearer $X_BEARER_TOKEN`, `query`, and `max_results`, then return posts that `posts_to_items` can map. That call is the operator's code. `UnconfiguredXAdapter.fetch` raises instead.

## Decision shape

Screen choices are `keep` and `drop`. The relevance score uses two levels, "Not relevant" then "Relevant hotspot". Fact-check choices are `supported`, `unsupported`, and `needs_human`. The support score uses three levels, "Unsupported", "Needs a human", then "Supported".

Those are the same kind of enumerated questions as `ticket_router` and `s1a decide`. `s1a decide` still answers one `choice`. It does not grow a `score` question. Score is asked only by `s1a/content/omni_client.py`, one short prefill per question, Public231-style: a string `state`, one entry in `questions`, and no images.

The offline decider is a lexical rule, the same role as the ticket router's keyword baseline. On the shipped fixture it matches the labels by construction. That is a wiring check, not a Jev or Omni quality result. The rule keeps a whole word such as `model` or `agent`. It also keeps "model trains", and it drops "supermodel", "chipotle", and "agency". The fixture simply does not include the "model trains" trap.

## Draft workers

Nothing in pull requests 164–166 has to be merged. The client needs a base URL that already speaks `POST /v1/systemone` and `GET /health`.

| Role | Worker | Draft PR | Request flag |
|---|---|---|---|
| Screen, and fact-check by default | StartLux Instant 4B (`StartLux-Decision-v1.1-4B-Instant`) | [system1-omni #164](https://github.com/InferMatrixLab/system1-omni/pull/164) | `think_mode: never`. The `thinking` object is not sent. |
| Optional heavier fact-check | PPLX decider (`perplexity-ai/pplx-decider-v1.1-27b`) | [system1-omni #166](https://github.com/InferMatrixLab/system1-omni/pull/166) | `thinking: {"enabled": false}` on `OMNI_FACTCHECK_BASE_URL`. |
| Not on this path | Clef / Clef-flash | [system1-omni #165](https://github.com/InferMatrixLab/system1-omni/pull/165) | Refused. |

`OMNI_MODEL` overrides the StartLux alias. `OMNI_FACTCHECK_MODEL` overrides the fact-check alias. PPLX is not assumed to share StartLux's port: `--fact-check-worker pplx` without `OMNI_FACTCHECK_BASE_URL` is an error. Clef is an error on either step.

With `OMNI_BASE_URL` unset, `--backend auto` is the lexical rule and opens no decision socket. `--backend omni` without that variable is a usage error.

## Human gates

Pass `--picks` (`{"ids": [...]}`) and `--publish` (`{"decisions": {"id": "publish"|"hold"}}`). On a terminal, omitting them prompts. Otherwise the process exits 2 and writes a packet only when `--packet` is set. The packet's `stopped_at` is `human_pick` or `human_publish`. It is not a finished run.

`--allow-unsafe-skip-gates` is unsafe. It invents a pick (the first three kept ids, or every kept id when fewer than three were kept) and holds every picked item. It does not publish. An explicit file or hook wins over the flag: the flag does not replace a decision you already passed.

A decision of `publish` is still only a record. There is no publisher, no CMS write, and no MCP tool that posts. Hosts that speak MCP should collect the human files and run this command. `run_agent` is the wrong tool: it finishes a loop, and this desk has to stop for a person.

Python hooks, for a caller that already has the person in the loop:

```python
from s1a.content.rules import RuleDecider
from s1a.content.workflow import run_workflow

run_workflow(items, decider=RuleDecider(), pick_hook=ask_for_ids, publish_hook=ask_publish_or_hold)
```

`pick_hook` sees kept items without labels. `publish_hook` sees the fact-check choice and score name. Both results are checked: 3 to 5 kept ids when at least three were kept, otherwise every kept id; publish decisions must name each picked id as `publish` or `hold`.

## Latency gate, not attached

Every report carries `latency_gate.attached: false`. Per-request client wall times are stored on Omni calls (`calls[].latency_ms`). Rule rows are omitted from the gate so a 0 ms baseline cannot clear a ceiling.

Hang a Public231-C1-style ceiling later, after a worker exists:

1. Serve the draft worker. Confirm `GET $OMNI_BASE_URL/health`.
2. Discard a warmup of 5 requests. The workflow itself does not drop them; run a throwaway command first, or strip those calls from the report.
3. Keep concurrency at 1. This client already sends one POST at a time.
4. Repeat three measured rounds on the same fixture.
5. Pool the successful request latencies and run:

```bash
python -m s1a.content latency-report \
  --report runs/recipes/content-workflow/report.json \
  --p50-max-ms <ceiling> \
  --p95-max-ms <ceiling> \
  --min-samples <measured-requests>
```

No ceiling is pinned. A p50 from Public231 on another checkpoint is not this fixture's budget. The clock here is `client_request_wall_ms` (POST start through the response body). System1-Omni's bench clock is `post_to_body_json_and_schema_validation`. Do not treat them as one number. With a ceiling set and too few Omni samples, the command fails closed and exits 1. With no ceiling, it prints the summary and exits 0.

The runnable fixture path is [the recipe](../recipes/content-workflow/README.md).
