# system1-agents

> [!NOTE]
> **Give your agents a System 1 decision model. Start from a prebuilt agent or build your own.**
>
> Describe the task. Claude Code or Codex runs a prebuilt agent or builds a new one, on [Jev](https://typesafe.ai),
> [Laya](https://huggingface.co/convaiinnovations/laya) or [Cua-S1 Nano](https://huggingface.co/cua-ai/cua-s1-nano-0.1).
> Browser use, computer use, robotics and games ship ready to run.
>
> **Up to 6× faster and 25× cheaper than a chat model, at the same score.**

[![test](https://github.com/ThinkFlowLab/system1-agents/actions/workflows/test.yml/badge.svg)](https://github.com/ThinkFlowLab/system1-agents/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![skill](https://img.shields.io/badge/skill-npx%20skills%20add-black.svg)](#from-claude-code-or-codex)

[Quickstart](#choose-your-path) · [From Claude Code or Codex](#from-claude-code-or-codex) · [Benchmarks](docs/benchmarks.md) · [Docs](#docs)

<!-- placeholder: name the chat model behind the "chat model" columns (the MODEL_NAME of these runs). -->

| scenario | Jev | chat model | speedup | Jev cost | chat model cost | more expensive |
|---|---|---|---|---|---|---|
| Browser use (Allrecipes)† | 35.7 s | 138.7 s | 3.89× | $0.091356 | $1.494331 | 16.4× |
| Custom agent (ticket router, 30 tickets) | 12.7 s | 65.3 s | 5.14× | $0.000764 | $0.014800 | 19.4× |
| Computer use (Windows Calculator) | 17.2 s | 30.1 s | 1.75× | $0.000447 | $0.003002 | 6.7× |
| Robotics (ALFWorld) | 7.4 s | 25.7 s | 3.47× | $0.000221 | $0.002800\* | 12.7× |
| Games (2048, 20 moves) | 27.1 s | 68.5 s | 2.53× | $0.000447 | $0.006139 | 13.7× |
| Games (Millionaire) | 15.3 s | 21.6 s | 1.41× | $0.000168 | $0.000892 | 5.3× |
| Games (Blackjack) | 2.3 s | 14.7 s | 6.39× | $0.000021 | $0.000531 | 25.3× |

† The first Allrecipes task of the [WebVoyager](https://github.com/MinorJerry/WebVoyager) task set
([He et al., 2024](https://arxiv.org/abs/2401.13919), Apache-2.0, attribution in [NOTICE](NOTICE)): a vegetarian
lasagna with over 100 reviews, 4.5 stars or more, for 6. The chat model of that row is Claude Fable 5.1 through
OpenRouter; both models pay it for the typed search text and the answer. \* Estimated; the chat-model run recorded no
cost. Each replay below is the episode behind its row, Jev on the left and the chat model on the right, both on the
wall clock. The other Allrecipes runs, longer games and the Google Flights driver comparison:
[docs/benchmarks.md](docs/benchmarks.md).

<table>
  <tr>
    <td width="50%"><img src="docs/assets/demos/allrecipes-comparison-8x.gif" alt="Allrecipes: a vegetarian lasagna search on the live site, Jev on the left, the chat model on the right, with Jev's probabilities over the page's controls under each step"><br><sub>Browser use, WebVoyager's Allrecipes task 0 on the live site, replay at 8× speed</sub></td>
    <td width="50%"><img src="docs/assets/demos/desktop-comparison-4x.gif" alt="Windows Calculator: clicks toward 12 times 7 until the display shows 84, Jev on the left, the chat model on the right"><br><sub>Computer use, the Windows Calculator, replay at 4× speed</sub></td>
  </tr>
  <tr>
    <td><img src="docs/assets/demos/ticket-router-comparison-8x.gif" alt="Ticket router: 30 labelled tickets routed to five queues, Jev on the left, the chat model on the right"><br><sub>Ticket router, 30 tickets to five queues, replay at 8× speed</sub></td>
    <td><img src="docs/assets/demos/alfworld-validated-comparison-2x.gif" alt="ALFWorld household task with its AI2-THOR scene, Jev on the left, the chat model on the right"><br><sub>ALFWorld, an embodied household task, replay at 2× speed</sub></td>
  </tr>
  <tr>
    <td><img src="docs/assets/demos/2048-comparison-8x.gif" alt="2048 over 20 moves, Jev on the left, the chat model on the right"><br><sub>2048, 20 moves, replay at 8× speed</sub></td>
    <td><img src="docs/assets/demos/millionaire-comparison-4x.gif" alt="Millionaire quiz ladder, Jev on the left, the chat model on the right"><br><sub>Millionaire, a 15-question quiz ladder, replay at 4× speed</sub></td>
  </tr>
  <tr>
    <td><img src="docs/assets/demos/blackjack-comparison-2x.gif" alt="One Blackjack hand, Jev on the left, the chat model on the right"><br><sub>Blackjack, one hand, replay at 2× speed</sub></td>
    <td></td>
  </tr>
</table>

## Choose your path

### From Claude Code or Codex

```bash
npx skills add ThinkFlowLab/system1-agents                                                              # the skill: Claude Code, Codex, Cursor
claude plugin marketplace add ThinkFlowLab/system1-agents && claude plugin install s1a@system1-agents   # plus the browser subagent and the MCP server
```

The skill tells the host when to hand a task to a System 1 agent and which command to run. The ticket router from
the table above, from Claude Code:

> Route this ticket to logistics, payment, returns, account or human: "I was charged twice for order 4411 and I
> want the second charge refunded."

The host runs one `s1a decide` over the five queues and reports the queue with its probability, in about 400 ms.
Page tasks, games and one-off selections go the same way; arithmetic, deduction and free text stay with the chat
model. The walkthrough, the plugin's keys and what to delegate: [docs/skills.md](docs/skills.md).

### From the command line

```bash
git clone https://github.com/ThinkFlowLab/system1-agents && cd system1-agents
uv sync && cp .env.example .env     # the first sync resolves the openjiuwen pin and takes a few minutes
```

Put a Jev key in `.env` (`TYPESAFE_API_KEY` from the [TypeSafe console](https://console.typesafe.ai), or
`OPENROUTER_API_KEY`), then ask for one decision and run one agent with each model:

```bash
uv run s1a decide --state '{"player_total": 18, "dealer_upcard": 9}' \
  --option hit="take a card" --option stand="keep the hand" --rules "stand on 17 or more"
uv sync --extra blackjack
uv run s1a run blackjack --model jev --rethink off --episodes 20
uv run s1a run blackjack --model llm --rethink off --episodes 20     # the chat model in the same agent
```

`decide` prints one JSON object with `choice`, a probability per option, `confidence` and `ms`; `run` writes a job
folder with the score. Without a key, `--model cua` answers in process after `uv sync --extra cua`.

### As an MCP server

```bash
codex mcp add s1a -- uv run --project /path/to/system1-agents s1a-mcp
```

It serves three tools, `list_agents`, `run_agent` and `decide`, to any host that speaks MCP.

## Build your own

A System 1 agent is one module under `s1a/agents/` that ends in a frozen `SPEC`; Blackjack is 111 lines. The builder
skill runs in Claude Code from this checkout, probes the task with 8 to 12 hand-written decisions before it writes
code, and stops when the task needs deduction or arithmetic:

> Build a System 1 agent for <task>.

The gates and the templates: [docs/skills.md](docs/skills.md#build-a-system-1-agent).

## What ships

- `allrecipes`: browser use, [WebVoyager](https://github.com/MinorJerry/WebVoyager)'s first Allrecipes task, headed on the live site.
- `flights`: browser use, a Google Flights search over `@playwright/mcp`.
- `desktop`: computer use, any Windows or macOS app window through [Cua Driver](https://cua.ai/docs/cua-driver).
- `ticket_router`: 30 labelled support tickets to five queues.
- `alfworld`: household tasks in text, with the AI2-THOR scene in the replays.
- `game2048`, `millionaire`, `blackjack`: games with a score per episode.
- `injection_guard`: a rail that answers one question at a hook of a running agent and fails closed.

Every agent runs on `jev`, `laya` or `cua`, and on the chat model for the comparison. Flags, run commands and
extras: [docs/agents.md](docs/agents.md).

## How it works

Each agent is a stock [openJiuwen](https://github.com/openJiuwen-ai/agent-core) agent with a System 1 decision
model as its `model`. On a decision turn the model gets the state and the options and answers with one of them;
planning, typed values and the final answer stay with the chat model in the same agent. Any decision model with that
interface fits: [docs/architecture.md](docs/architecture.md), [docs/decision-models.md](docs/decision-models.md).

## Docs

- [docs/benchmarks.md](docs/benchmarks.md): the six runs above, the Google Flights driver comparison, a longer game, the guard rail.
- [docs/skills.md](docs/skills.md): the caller skill, the builder skill, what to delegate.
- [docs/agents.md](docs/agents.md): every agent with its flags, run command and extra.
- [docs/architecture.md](docs/architecture.md) and [docs/decision-models.md](docs/decision-models.md): the fronts, the model slot, the model interface, adding a backend.
- [docs/browser-front.md](docs/browser-front.md): the browser policy, decision by decision.
- [docs/configuration.md](docs/configuration.md): environment variables, defaults and reader subsystems in one table.
- [docs/glossary.md](docs/glossary.md): terms the documentation glosses on first mention.
- [docs/why.md](docs/why.md): the problem, the philosophy, the precedents.
- [docs/roadmap.md](docs/roadmap.md) and [CHANGELOG.md](CHANGELOG.md).

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) has the dev install, the checks and the hooks. Apache-2.0.
