# The three fronts and their specs

Every agent states every field. The spec classes in `s1a/spec.py` define no defaults.

## Tool front: `ToolAgentSpec`, template `_templates/tool_agent.py`

A DeepAgent plays through two tools, `observe` and `act`, with a System 1 decision model in the model slot
(`--model jev|llm|random|rule|laya|cua`).

| field | meaning |
|---|---|
| `name` | the module name: lowercase letters, digits and `_` after a letter; also the job folder name |
| `description` | one sentence, shown by `list_agents` and the caller skill |
| `rules` | the text the model reads on every decision |
| `budget` | `Budget(max_steps, timeout_s, stall_after)`; the defaults of `--max-steps` and `--timeout` |
| `flags` | `(ArgumentParser) -> None`: the agent's own switches, after the shared ones |
| `series` | `(Namespace) -> Series`: the seeds, `env_for(seed)`, the page `session`, the `baseline`, `annotate` |

The `Env` an agent adapts: `reset`, `observe`, `candidates`, `step(key)`, `done`, `score` (`s1a/env.py`).
Page-backed games open their page through `BrowserHands` from `s1a/tool/hands.py` in `series.session`.
Desktop agents bind one native window through Cua Driver: `WindowEnv` from `s1a/desktop/env.py` lists the
window's clickable elements as candidates, plus `done` and `abstain`, and `s1a/desktop/driver.py` opens the
`cua-driver mcp` process in `series.session`; `agents/desktop.py` is the generic agent (`--app`, `--goal`, `--expect`).

## Browser front: `BrowserAgentSpec`, template `_templates/browser_agent.py`

openJiuwen's browser subagent with `BrowserDecisionModel` in the slot (`--model jev` for TypeSafe Jev, `laya` for
Laya in process, `cua` for Cua-S1 Nano in process, `llm` for the chat model alone); the chat model types values and writes the answer.

| field | meaning |
|---|---|
| `name` | the module name: lowercase letters, digits and `_` after a letter |
| `description` | one sentence, shown by `list_agents` and the caller skill |
| `rules` | the operation rules sent with every decision; one string per site family |
| `language` | `en` or `cn`: picks the target, value and answer rule tables |
| `budget` | `max_steps` is the subagent's iteration cap, the default of `--max-steps`; `timeout_s` the default of `--timeout`; `stall_after` the actions without a page change before BLOCKED |
| `goal` | a fixed task string, or `None` when every call brings `--goal` |

The policy switches are run-time flags, the same for every browser agent: `--batch on` sends
each action and the next probe as one `browser_run_code_unsafe` call (no target validation, needs `unsafe_dev`);
`--prefetch off` stops the typed-value calls that start as soon as a probe shows an editable field;
`--goal-values on` sends values extracted from the goal to Jev as a choice head.

## Rail front: `RailSpec`, template `_templates/rail.py`

One Jev question at one callback hook of a DeepAgent, acted on above a threshold.

| field | meaning |
|---|---|
| `hook` | `AgentCallbackEvent.BEFORE_TOOL_CALL` or `AFTER_TOOL_CALL`; at both `ctx.inputs` is a `ToolCallInputs` (`tool_name`, `tool_args`, and after the call `tool_result`, `tool_msg`) |
| `question` | `noul` (a yes-or-no statement) or `choice` |
| `rules`, `criteria`, `flagged` | the question, what each answer means, the key whose probability is banded |
| `state_of` | `(AgentCallbackContext) -> dict or None`: what Jev sees; `None` skips the event |
| `thresholds` | `Thresholds(allow, act)`: at or below `allow` nothing, at or above `act` the action runs |
| `act` | `async (ctx, verdict) -> None`: quarantine a result, block a call, log, escalate |
| `labelled_set` | JSONL of `{"state", "label"}` records; `s1a run <rail>` reports precision and recall |

A rail attaches to an agent with `create_deep_agent(rails=[rails.DecisionModelRail(SPEC, build_model("jev"))])`.
An `after_tool_call`
rail may rewrite `ctx.inputs.tool_result` and `ctx.inputs.tool_msg`; the injection guard does.
