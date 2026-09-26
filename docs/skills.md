# The two skills: call a System 1 agent, build a System 1 agent

## Call one from Claude Code, Codex, Cursor or Hermes

The caller skill, `skills/s1a/SKILL.md`, tells the host when to delegate and which command to run. It follows the
agentskills.io layout. Codex also reads it through the `.agents/skills/` symlink, and Hermes takes the same directory
into `~/.hermes/skills`. The Claude Code plugin adds the `s1a-browser` subagent and the MCP server.

```bash
npx skills add ThinkFlowLab/system1-agents
claude plugin marketplace add ThinkFlowLab/system1-agents && claude plugin install s1a@system1-agents
```

The plugin's MCP server starts as `uv run --project <plugin root> s1a-mcp`, with `uv` on `PATH`; the first start runs
`uv sync` in the plugin folder and takes minutes to resolve the `openjiuwen` git pin. Export `TYPESAFE_API_KEY` (or
`OPENROUTER_API_KEY`) in the shell that launches Claude Code, since the plugin folder has no `.env`. Runs and logs
land under the plugin folder unless `S1A_HOME` names another root ([configuration.md](configuration.md)). Every host runs the command from a checkout of
this repository.

### A ticket through the skill

With the skill installed, give the host a routing task:

> Route this ticket to logistics, payment, returns, account or human: "I was charged twice for order 4411 and I
> want the second charge refunded."

The skill matches the case it lists for one selection over a fixed option list and runs `s1a decide` with the
ticket as the state, the five queues as the options and a short form of the shipped ticket agent's rules:

```bash
uv run s1a decide \
  --state '{"title": "Charged twice", "description": "I was charged twice for order 4411 and I want the second charge refunded.", "order_status": "delivered"}' \
  --option logistics="delivery tracking, delivery progress or delivery problems" \
  --option payment="charges, failed payments or duplicate payments" \
  --option returns="requests for returns, exchanges or refunds" \
  --option account="login or account access problems" \
  --option human="insufficient information, several independent requests, or an explicit request for a human" \
  --rules "Route the current explicit request to exactly one queue. A payment problem with an explicit request for a refund belongs to returns. If no unique queue fits, choose human."
```

One JSON object comes back in about 400 ms: `choice`, a probability per queue, `confidence` and `ms`. The shipped
`ticket_router` agent uses the same queues and rules: it routes a seeded batch of 30 labelled tickets and scores the
correct routes. The first row of the README's table is one such batch with each model:

```bash
uv run s1a run ticket_router --model jev --rethink off --episodes 1
uv run s1a run ticket_router --model llm --rethink off --episodes 1
```

A page task goes the same way. The prompt names the site, the values to enter and the stop condition; the skill
runs `s1a run flights --model jev --goal "..."` and reads the answer from `final` in the JSON.

### What to delegate

- Delegate a sequence of selections over visible controls (search forms, filters, date pickers, result lists), a
  game or environment that enumerates its legal moves each step, or one selection over a fixed option list: routing,
  ranking, gating, "is this text an instruction". On 48 hand-written single-step decisions Jev answered 45 right.
- Keep arithmetic, constraint deduction, search over move trees, free-text generation and any value the page never
  shows with the chat model. The three probe misses, Minesweeper, Wordle and a severity rubric, needed deduction. A
  plain page fetch needs no agent; use `curl`.

### What comes back

Every `run` prints one JSON object on stdout and nothing else there; the harness logs go to files under
`runs/logs`. A browser agent's object has `final`, the answer. A tool agent's object is the series summary with its
`job_dir`. Flags, exit codes and the job-folder layout: [architecture.md](architecture.md).

### The MCP server on its own

```bash
codex mcp add s1a -- uv run --project /path/to/system1-agents s1a-mcp
```

`s1a-mcp` serves the same agents over stdio as three tools. `list_agents()` returns every agent with its front, its
description and the flags `run_agent` accepts for it; an agent whose optional dependency is missing is listed as
unavailable with the error. `run_agent(name, flags)` runs one agent with the flags of `s1a run <name>` and returns
its JSON object. `decide(state, options, rules, model="jev")` answers one choice question: the chosen key, a
probability per option, a confidence and the latency in ms. `model` accepts `jev`, `laya` or `cua`; callers that
omit it keep using Jev. For local decisions, install the matching extra in the server's checkout (`uv sync
--extra laya` or `uv sync --extra cua`) and pass `model="laya"` or `model="cua"`; no Jev API key is needed.
The first local call may download the checkpoint. Each call loads and closes its model; `ms` measures the
decision, not model loading. Agent runs and decisions are serialized, and model output stays off the stdio
protocol stream.

## Build a System 1 agent

A System 1 agent is one module under `s1a/agents/` that ends in a frozen `SPEC`; Blackjack is 111 lines. The loop,
the decision-model layer, the rethink rail, the job folders, the CLI and the MCP server are shared; they find a new
module by name. The builder skill, `.claude/skills/build-s1a-agent/`, runs in Claude Code from this checkout:

> Build a System 1 agent for <task>.

It produces the module, its test and a row in the agents table, and stops at the first gate that fails:

1. Intake: the task, where the state comes from, how the options are enumerated each step, the score, and whether
   any step needs arithmetic, deduction, search or generated text.
2. Fit probe before any code: 8 to 12 hand-written decisions through `s1a probe cases.jsonl`. Under 80 percent
   right, or one miss that needed deduction, the verdict is "not a decision-model task".
3. Front: a tool loop for an environment that enumerates moves and scores, a browser policy for a page with visible
   controls, a rail for one question at a hook of a running agent.
4. Scaffold from the front's template under `s1a/agents/_templates/`, with the state-design rules from the skill's
   references.
5. Verify one rung at a time: the offline test, then `--model random`, `rule`, `jev` and `llm` on the same seeds, then
   the results table and the full suite.
