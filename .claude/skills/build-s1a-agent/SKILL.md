---
name: build-s1a-agent
description: Build a new System 1 agent (an openJiuwen agent with a System 1 decision model in its model slot) for a task the user names, in this repository. Runs a fit probe first, then scaffolds one agent module from the template of the right front (tool, browser or rail), its test and its README row, and verifies it model by model. Use when the user asks to build, add or scaffold a System 1 agent, an S1A agent, a Jev agent, a new game, task, site or rail for s1a.
---

# Build a System 1 agent

A System 1 agent is one module under `s1a/agents/` that ends in a frozen `SPEC`. The loop, the decision-model layer, the
rethink rail, the job folders and the CLI are shared and never change for a new agent. This skill produces the
module, its test and a README row, and stops at the first gate that fails.

Read `references/fit-rule.md` before the intake and `references/state-design.md` before the scaffold. The three
fronts and every spec field are in `references/fronts.md`.

## 1. Intake, one message

Ask for, or confirm from the request, these five facts. Stop until they are clear.

1. The task, in one sentence, and what a finished episode looks like.
2. Where the state comes from: an API or library, a page, a text environment, a callback in a running agent.
3. How the options are enumerated at each step: the library's legal moves, the page's controls, a fixed list.
4. The score: what counts, and whether a baseline exists (a rule, an expert plan, a published number).
5. Whether any step needs arithmetic, constraint deduction, search, or generated text. One yes is a stop.

## 2. Fit gate, before any code

Write 8 to 12 hand-written cases as JSONL, one decision each, spread over the task's situations, and run them:

```bash
uv run s1a probe cases.jsonl
```

Each line: `{"state": {...}, "options": {"key": "description"}, "rules": "...", "accept": ["key"], "note": "..."}`.
The verdict prints as `fits` at 80 percent right or above over at least 8 cases, `too few cases` under 8, otherwise
`not a decision-model task`. A wrong case that needed deduction or arithmetic is a stop even when the total passes. Report the
table to the user. On a stop, say which cases failed and why, name the nearest task that would fit, and end.

## 3. Front

- Tool front when a library or environment enumerates the moves and reports a score: `ToolAgentSpec`.
- Browser front when the task is a page with visible controls: `BrowserAgentSpec`.
- Rail front when the task is one yes-or-no or one selection at a hook of a running agent: `RailSpec`.

## 4. Scaffold

Copy the front's template from `s1a/agents/_templates/` to `s1a/agents/<name>.py` and replace every
part; the template's comments say what each part is. `SPEC.name` is the module name. Write
`tests/test_agents_<name>.py` with the two tool-front classes of `tests/test_templates.py`, `TestNimEnv` and
`TestNimThroughTheLoop`, rewritten for the new module: the env's reset, candidates and winning line, then the rule
and random models through `series.play` with `loop.WORKSPACE` and `series.optional_chat_model` patched as there.
For a browser agent, copy `TestBrowserTemplateOffline`: the spec reaches the faked subagent through
`support.browse_offline`. For a rail, copy `TestRailTemplateOffline`: precision and recall on five hand-labelled
records through a `ScriptedModel(noul=[...])` from `s1a.decision_models`. Add one row to the agents table in `README.md`. Follow `references/state-design.md` for the
observation, the candidate keys, the rules text and the budget. Do not touch `s1a/tool/`,
`s1a/browser/`, `s1a/rails.py` or `s1a/spec.py`.

## 5. Verify, one rung at a time

Run each command, read its output, fix the agent before the next rung. Stop at the first rung that fails.

```bash
uv run pytest tests/test_agents_<name>.py -q                      # the adapter contract, no keys
uv run s1a run <name> --model random --rethink off --episodes 3   # mechanics through the loop, no keys
uv run s1a run <name> --model rule --rethink off --episodes 3     # when a baseline exists
uv run s1a run <name> --model jev --rethink off --episodes 3 --log   # keys: latency, invalid keys must be 0
uv run s1a run <name> --model llm --rethink off --episodes 3      # the same seeds with the chat model
uv run python -m evals.table evals/results                             # one row per model
uv run pytest tests -q                                                 # the whole suite stays green
```

Every `run` prints one JSON object: the series summary with `scored`, the episodes that got a score, `errors`, the
count the model could not play, and `job_dir`. `random` and `rule` exist for the tool front only; `laya`
(Laya in process, after `uv sync --extra laya`) runs wherever `jev` does, and so does `cua` (Cua-S1 Nano, after
`uv sync --extra cua`) except on a rail. For a browser or rail agent the
key-free rung is the offline test from step 4. The paid rung follows. A browser agent runs one task per call and
needs the chat-model key and a Jev key: `uv run s1a run <name> --model jev --goal "..."`. A rail runs its
labelled set and needs a Jev key: `uv run s1a run <name> --labelled-set records.jsonl`.

Report the table, then prepare the demo/evidence handoff below. Series of a hundred episodes cost money;
ask the user before starting one.

## 6. Demo and handoff

For a reusable use case, add a recipe using [the recipe template](../../../recipes/TEMPLATE.md) and
[add-agent-recipe skill](../../../.agents/skills/add-agent-recipe/SKILL.md), then update the recipe index.
Follow [CONTRIBUTING.md's video guide](../../../CONTRIBUTING.md#agent-video-demos)
for the required video of the application/task, the System1-Agents decision-model agent and System1-Omni
inference in the same run. Show actual input, selected/executed actions, engine identity and checked result.
Choose an application from the guide's README candidates that exercises the new agent's actual behavior.
Use a terminal recording for a text agent or rail; label scripted checks and recorded replays.
Use a supported System1-Omni path; record missing support or integration as a review gap. Follow PR #35's
worked example linked in the guide. Keep an important PR draft until its video is supplied or a maintainer
accepts the documented exception.
Prepare the PR's **Demo / evidence** section with the clip, reproduction command and linked run artifacts.
Recording, model execution and uploading each require the authorization/resources already established for
the task. If any are unavailable, report the gap and hand off the prepared instructions.
