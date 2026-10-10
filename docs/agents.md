# Agents: what each one measures and how to run it

Four kinds of agent share the loop, the models and the job folders: browser use over `@playwright/mcp`, computer use
over [Cua Driver](https://cua.ai/docs/cua-driver), games and embodied text, and rails that answer one question at a
hook of a running agent. The injection guard rail fails closed: a decision error quarantines the tool result too.

| agent | front | what it measures | run |
|---|---|---|---|
| `allrecipes` | browser | browser use against the chat model: WebVoyager task Allrecipes--0, a vegetarian lasagna with over 100 reviews, 4.5 stars or more, for 6 | `s1a run allrecipes --model jev --batch on --headed` |
| `flights` | browser | browser use: wall clock on a Google Flights search | `s1a run flights --model jev --batch on --profile-out run.json` |
| `desktop` | tool | computer use: clicks toward `--goal` until the window shows `--expect`; the Calculator on macOS or Windows is the example | `s1a run desktop --app Calculator --goal "compute 12 times 7" --expect 84 --execute --clear "All Clear" --model jev --rethink off --episodes 1` |
| `ticket_router` | tool | correct routes on a seeded batch of 30 labelled tickets, five queues | `s1a run ticket_router --model jev --rethink off --episodes 1` |
| `alfworld` | tool | success on unseen household tasks in text | `s1a run alfworld --model jev --rethink on --episodes 12 --stride 11` |
| `game2048` | tool | score and largest tile at a move cap | `s1a run game2048 --model jev --rethink on --episodes 10` |
| `millionaire` | tool | winnings on a 15-question quiz ladder | `s1a run millionaire --model jev --rethink off --episodes 5` |
| `blackjack` | tool | payoff per hand (RLCard) | `s1a run blackjack --model jev --rethink off --episodes 100` |
| `sokoban` | tool | solved fraction on 100 selected Valen levels: text boards over `jev`; `--visual` renders for an image-reading model | `s1a run sokoban --model jev --rethink off --episodes 100` |
| `injection_guard` | rail | precision and recall on a labelled injection set | `s1a run injection_guard` |

Every tool agent takes `--model jev|clm|laya|cua|llm|random|rule`, `--rethink on|off`, `--episodes N`, `--seed S`,
`--max-steps` and `--timeout`, and writes a Harbor-shaped job folder under `evals/results/<agent>/`. A browser agent
takes `--model jev|clm|laya|cua|omnijev|llm` and `--goal`. A rail takes `--model jev|clm|laya|laya-served`, the models that answer `noul`.
`uv run python -m evals.table evals/results` aggregates every job folder per eval and model into one table.

Every `run` prints one JSON object on stdout and nothing else there; `s1a-mcp` serves the same agents over stdio
with `list_agents`, `run_agent` and `decide`. Flags, exit codes and the job-folder layout:
[architecture.md](architecture.md). The extras each agent needs and the keys: `CONTRIBUTING.md`. The `--model` values and the
models behind them: [architecture.md](architecture.md#models).

## Agent-specific flags

`s1a run <agent> --help` lists every flag with its default. Beyond the shared ones: `flights` and `allrecipes` take `--goal`,
`--batch on|off`, `--prefetch on|off`, `--goal-values on|off`, `--profile-out` and `--logs-dir`; `desktop` takes
`--app`, `--app-path`, `--window-title`, `--goal`, `--expect`, `--execute`, `--plan`, `--clear`, `--text`,
`--text-target`, `--text-mode`, `--verify-file` and `--pixel-target`; `ticket_router` takes `--dataset` and
`--batch-size`; `injection_guard` takes `--labelled-set`. The four games take no flag of their own.

### Desktop text input

`--text` supplies the content for a `type:*` action. `--text-target` selects an exact field label or native
identifier. The default `--text-mode insert` inserts at the current selection; `replace` sets the whole field.
Both require a fresh readback before the input is recorded as successful. Chinese and multiline text are
supported through native field replacement. `--window-title` selects the document when an app has several windows.

`--verify-file` requires the expected window state and a file changed during this episode whose UTF-8 content
matches `--text`. Without `--execute`, the first decision is only recorded as a plan.
The [macOS fixture](../evals/desktop/README.md) provides a local Laya demo and a fixed-plan execution check.

### Desktop screenshot targets

`--pixel-target KEY=X,Y` offers named points in screenshot fractions. Local Cua-S1 4B multimodal can select
among them when controls have no accessibility elements. Each click carries its screenshot capture ID.
Install the `cua-four-b` extra and set `CUA_S1_VARIANT=4b`, `CUA_S1_MODALITY=multimodal`.
See [the macOS visual task](../evals/desktop/visual.md) for setup and verification commands.

## Allrecipes

`allrecipes` is the first Allrecipes task of the [WebVoyager](https://github.com/MinorJerry/WebVoyager) task set
([He et al., 2024](https://arxiv.org/abs/2401.13919), Apache License 2.0, attribution in `NOTICE`), verbatim: find a
vegetarian lasagna with more than 100 reviews, a rating of at least 4.5 stars, suitable for 6 people, and answer in
text. The run is headed
because the site answers a headless Chromium with a bot wall. `scripts/browser_showcase.sh allrecipes 2` runs both models
twice with a frame after every browser call and renders the pair GIFs from the median run of each model; the numbers
and the replay are in [benchmarks.md](benchmarks.md#browser-use-allrecipes).

## The ticket router

`ticket_router` routes locally labelled tickets to five queues: logistics, payment, returns, account and human. The
shipped set is 30 tickets in `s1a/agents/_data/ticket_router_eval.jsonl`, each with a title, a description and an
order status; the label is read by the scorer alone. `--dataset` points the agent at a
JSONL of your own with the same fields, and `--batch-size` caps the tickets per episode. The rules text the model
reads is `RULES` in `s1a/agents/ticket_router.py`.

## Sokoban

`uv run s1a run sokoban --model jev --rethink off --episodes 100` plays the bundled 100 Valen levels as text
boards. No extra is needed. `--seed` is the zero-based level offset; the selected range must fit within 100
levels. The score is 1 for solved and 0 otherwise. Use `--model random` for an offline smoke run; there is no
rule baseline. Rethink must be off. See [protocol and provenance](../evals/sokoban/README.md).

`--visual` plays the same levels over Valen's own rendering — one PNG per decision, theme and tile size from
the level, rules and direction labels in the level's language — with the ASCII board left out. Text-only
backends (`jev`, `clm`, `laya`, `laya-served`, text-modality `cua`, `llm`) are refused before the first
decision instead of running blind. `cua`'s multimodal 4B takes the picture, but admission is capability,
not a result: no visual Sokoban run is recorded for it, so the `jev` command above stays the reference
text-mode run. (`omnijev` is image-capable too, but it is a browser-front choice; `s1a run` cannot
select it.) `--model random --visual` smokes the rendering offline (`uv run --extra visual`);
the served Valen backend this observation was made for ships as its own change.
