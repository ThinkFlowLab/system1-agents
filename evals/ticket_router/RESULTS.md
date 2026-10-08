# English ticket router validation

Date: 2026-09-22. Local synthetic tickets only; no customer system is connected.

## Dataset and scope

- Twelve fit-probe cases and thirty separate evaluation tickets, six per queue.
- All titles, descriptions, candidate descriptions, rules, baseline keywords and test fixtures are in English.
- Translated from the earlier Chinese fixtures without changing labels, order statuses or scenario categories.
- Evaluation IDs are derived from the English title and description, not from the expected label.
- Seeds 0, 1 and 2 shuffle the same thirty tickets. Ninety decisions are still thirty independent samples.
- Observations exclude labels and evaluation metadata; rethink is off and stall_after is zero.
- The keyword baseline is case-insensitive and matches whole words/phrases. Multiple or no category matches route to human.
- This is an offline, labelled evaluation environment, not a production connector for unlabelled tickets.

Evaluation SHA-256: `18608fa642c5d40d665003cba9318dca1593894d1e17b31b91fbd54fcfc37062`.

Probe SHA-256: `f5af40ce093a4c4fea9e4c5094b27caa08dc9fc87cbc574a3d948f4000170da2`.

## Current validation

| Strategy | Correct / planned | Coverage | Errors | Invalid actions | Total episode seconds | API cost USD |
|---|---:|---:|---:|---:|---:|---:|
| random | 17/90 | 100% | 0 | 0 | 2.6 | 0.0 |
| rule | 51/90 | 100% | 0 | 0 | 2.7 | 0.0 |
| Jev | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| LLM | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

The English fit probe was attempted but failed with `decisions connection failed`; it did not produce a fit verdict.
Jev and LLM batch validation was not started after that failed gate. English model quality remains unvalidated.
Earlier Chinese-language model results must not be attributed to this English dataset.

The random and rule runs have matching batch fingerprints for every seed. Accuracy uses the whole planned batch;
unprocessed tickets count as incorrect, and coverage is reported separately. Per-class recalls and per-ticket
predictions are in [validation.json](validation.json) and [the ticket listing](ALL_TICKETS_AND_RESULTS.md).

The thirteen ticket-router tests and seventeen subtests passed. They cover label isolation, reset/seeding,
invalid actions, complete/partial scoring and integration through the shared loop. These tests do not validate model quality.

## Reproduce

From the repository checkout after installing the core and dev dependencies:

```sh
uv run --no-sync pytest tests/test_agents_ticket_router.py -q
uv run --no-sync s1a probe evals/ticket_router/probe.jsonl
uv run --no-sync s1a run ticket_router --model random --rethink off --episodes 3 --seed 0
uv run --no-sync s1a run ticket_router --model rule --rethink off --episodes 3 --seed 0
```

Once the English probe passes (at least 10/12), run matched small model comparisons:

```sh
uv run --no-sync s1a run ticket_router --model jev --rethink off --episodes 3 --seed 0 --log
uv run --no-sync s1a run ticket_router --model llm --rethink off --episodes 3 --seed 0
```

Each episode is a batch, not one ticket. The default batch contains thirty tickets. The model commands require
configured keys; the LLM command also requires a chat-model name. Do not print credentials.

## Local source jobs

- random: `evals/results/ticket_router/2026-09-22__13-22-45-983908__random`
- rule: `evals/results/ticket_router/2026-09-22__13-23-05-526774__keywords`

Raw jobs are local ignored artifacts. The committed validation JSON contains the relevant baseline metrics
and routes so these claims do not depend on those local directories. Episode time excludes process startup.

The random/rule results were rerun after migration to the s1a package on base e4815e888d1dc477713ce99b88ba22164a9ddde2.
The earlier connection-failed model probe was not rerun during this migration.
