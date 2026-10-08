---
name: s1a
description: Delegate a page task with enumerable controls, a game or a quiz to S1A, an openJiuwen agent with a System 1 decision model in its model slot, through the s1a command; or ask the decision model for one fast selection over options you enumerate. Use for click-through web tasks with no arithmetic, and for routing, ranking or gating over a fixed option list. Not for a plain fetch (use curl), arithmetic, constraint puzzles or free-text negotiation.
---

# s1a

S1A runs openJiuwen agents with a System 1 decision model in the model slot: TypeSafe Jev over HTTP by default, Laya or
Cua-S1 in process. About 400 ms per request, one selection over the options the page or the game enumerates, a
probability per option. The command is `s1a`,
run from a checkout of https://github.com/ThinkFlowLab/system1-agents: `uv run --project <checkout> s1a ...`. Inside
the Claude Code plugin the checkout is `${CLAUDE_PLUGIN_ROOT}`. `python -m s1a` from the checkout is the
same command.

## When to delegate

- A web task that is a sequence of selections over visible controls: search forms, filters, date pickers, result
  lists, quizzes. The values to type come from the task text.
- A game or environment that enumerates its legal moves each step and reports a score.
- One selection over options you already hold, where a probability per option helps: routing, ranking, gating.

Not for: a plain page fetch (use curl), arithmetic, constraint puzzles, search over move trees, free-text
generation, or a task that needs a value the page never shows.

## Commands

```bash
s1a list
s1a run flights --model jev --goal "<site URL first, the values to enter, the stop condition>"
s1a run <agent> --model jev --rethink off --episodes 3
s1a decide --state '<JSON object>' --option key="what it means" --option other="what it means" --rules "<facts>"
s1a decide --state '{"title": "Charged twice", "description": "I was charged twice for order 4411 and I want the second charge refunded.", "order_status": "delivered"}' \
  --option logistics="delivery tracking, delivery progress or delivery problems" --option payment="charges, failed payments or duplicate payments" \
  --option returns="requests for returns, exchanges or refunds" --option account="login or account access problems" \
  --option human="insufficient information, several independent requests, or an explicit request for a human" \
  --rules "Route the current explicit request to exactly one queue. A payment problem with an explicit request for a refund belongs to returns. If no unique queue fits, choose human."
s1a run ticket_router --model jev --rethink off --episodes 1      # the shipped router: 30 labelled tickets, five queues, scores the correct routes
```

Every `run` prints one JSON object on stdout and nothing else there. The harness logs go to files under the
checkout's `runs/logs`. A browser agent's object has `ok`, `final`, `error` and `usage` with both models; `final` is
the answer. With `--model jev` it also has `report` and, when present (absent on a timeout), `status` and
`terminal`: `terminal.url` and `terminal.title` are where the answer was read. With `--model llm` it has `browser_result`, the subagent's own
verdict; `status`, `report` and `terminal` are absent there. A task takes seconds
to a few minutes; `--timeout` sets the wall clock (the agent's own default, 180 s for `flights`) and `--headed` shows the browser. A tool
agent's object is the series summary with `job_dir`, the job folder it wrote. `decide` prints `choice`,
`probabilities`, `confidence` and `ms`. Exit codes: 0 for a finished run, including one whose JSON has `ok: false`
and an `error`; 1 for a run, key or model error, one line on stderr; 2 for a usage error (an unknown agent, bad
flags, a malformed `--state`, `--option` or `@file`).
`--model jev` is TypeSafe Jev; `--model laya` is Laya, an open-weight System 1 decision model that runs in process after
`uv sync --extra laya`, with the same outputs and no Jev key; `--model cua` is Cua-S1 Nano, a small option scorer in
process after `uv sync --extra cua`, a baseline; tool agents also take `llm`, `random` and `rule`.

## Keys

`TYPESAFE_API_KEY` for Jev, or `OPENROUTER_API_KEY` for the proxy; `OPENAI_API_KEY` or `LLM_API_KEY`, the base URL
and `MODEL_NAME` for the chat model that types values and writes the answer. Export them in the host's environment.
The command also reads the `.env` of its checkout. Exported variables win over the file.
