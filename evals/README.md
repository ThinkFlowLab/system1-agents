# evals

Every evaluation runs through openJiuwen's DeepAgent with a System 1 decision model in the model slot. Each run writes a
Harbor-shaped job folder under `evals/results/<eval>/` (`config.json`, `result.json`, `summary.json`, one trial
folder per episode). `uv run python -m evals.table evals/results` aggregates the folders into one table.

| eval | measures | run |
|---|---|---|
| Blackjack (RLCard) | payoff per hand | `s1a run blackjack --model jev --rethink off --episodes 100` |
| 2048 (the MIT game, self-hosted) | score and largest tile at a move cap | `s1a run game2048 --model jev --rethink on --episodes 10` |
| Millionaire (self-hosted quiz, Open Trivia DB) | winnings; the 50:50 lifeline is a candidate the model may pick; six ladders: `--seed` plus `--episodes` stays at or below 6 | `s1a run millionaire --model jev --rethink off --episodes 5` |
| ALFWorld text (TextWorld) | success on unseen games | `s1a run alfworld --model jev --rethink on --episodes 12 --stride 11` |
| Google Flights timing | wall clock against browser-use/jev-ultrafast | `s1a run flights --model jev --batch on`; numbers in `docs/benchmarks.md` |
| Desktop (Cua Driver) | clicks toward `--goal` until the window shows `--expect` | `s1a run desktop --app Calculator --goal "compute 12 times 7" --expect 84 --execute --model jev --rethink off --episodes 1` |
| Ticket router (30 local labelled tickets) | correct routes to five queues | `s1a run ticket_router --model jev --rethink off --episodes 1` |
| Injection guard (rail) | precision and recall on a labelled set | `s1a run injection_guard` |

## The loop

`s1a/tool/loop.py` builds one DeepAgent per episode through `create_deep_agent`, with two tools per environment.
`observe` returns `{state, candidates, done, score}`; `act(key)` plays one candidate key and returns the same
shape. The model slot holds `ToolDecisionModel` (`s1a/tool/models.py`) over one of five decision models, or the
chat model (`--model`):

- `jev`: the model reads the environment, asks Jev one choice question, and answers with one `act` call.
- `laya`: the same, with Laya deciding in process (`uv sync --extra laya`); its tokens are free and unpriced.
- `cua`: the same, with Cua-S1 Nano deciding in process (`uv sync --extra cua`); a baseline that reads 256 bytes of state.
- `llm`: the chat model, which reads the candidates from the tool results.
- `random`: uniform over the candidates, the loop-overhead arm.
- `rule`: the hand-written baseline where a game has one (basic strategy, the fixed 2048 order, the ALFWorld
  expert plan).

`--rethink on` adds `RethinkRail` (`s1a/tool/rethink.py`) after every `act`. Three layers, cheapest first:
candidate hygiene in the adapters (2048 drops moves that did nothing, ALFWorld drops examine, look and
inventory); an exact repeat, the same key three times without a score change, blocks that key for one turn; a
semantic stall, six acts in 2048 or eight in ALFWorld without a score change and with a state seen before, asks
the chat model for a 60-word plan that the model reads in its state on the next turn. A fourth stall after three plans
without a score change ends the episode. Blackjack and Millionaire never stall and ignore the flag. Because three
`act(LEFT)` calls in a row are a legal 2048 line, the harness's own anomaly rail (its bailout ends a run on the third
identical call) stays off for these agents.

Every episode records its ticks (key, confidence, probabilities, latency, tokens, whether a plan was in the
state) and its rethink events in `agent/episode.json`; the count of chat-model calls and their token sums go to
`result.json`. For `llm` the decisions are the chat calls that produced an `act`; a tick whose key was
not offered is kept with `accepted: false`, counted as an `invalid_key`, and skipped by the replay. Its iteration
cap is twice the game's budget, since the chat model spends turns on unknown keys.
A run that reaches the iteration cap has `result_type: error` in `extra`; its score still counts. An episode that
raised (a decisions failure, an env error) keeps `error` set: `summary.json` reports it under `errors`, computes
the score statistics over the `scored` episodes only, and `evals.table` skips trials whose `result.json` has
`exception_info`.

## Setup

`uv sync`, plus `--extra blackjack` and `--extra alfworld` for those games (the README lists every extra), then
a `.env` with `TYPESAFE_API_KEY` (or `OPENROUTER_API_KEY`) for Jev, and `MODEL_NAME` with provider credentials for the chat model.
ALFWorld needs `uv sync --extra alfworld` and `ALFWORLD_DATA` in a Python 3.11 environment; the data comes from
`python scripts/alfworld-download` in a clone of alfworld/alfworld. The DeepAgent's workspace files land under `runs/evals/`.
For the complete reference table of all environment variables, provider endpoints, and defaults, see [docs/configuration.md](../docs/configuration.md).

## Protocol

Every model plays seeds `S` to `S + N - 1` (`--seed S --episodes N`) and meets the same deals and games.
`summary.json` holds the mean score with a 95 %
bootstrap interval, wins and losses, mean steps, the median decision latency and the rethink count. Quote
nothing below ten episodes; Blackjack wants 500 hands. The four measurements: `jev` against `llm` on the same
seeds (same loop, swap the brain); `jev` with `--rethink on` against `off`; the `random` model's wall clock per
act against a bare loop (loop overhead); and, later, Jev's top probability against the ALFWorld expert plan.
`summary.json` also holds decisions, chat calls, tokens (`chat_input_tokens`, `chat_output_tokens`,
`chat_cache_tokens`) and `cost_usd` (Jev at $0.042 per M input tokens; the chat model at OpenRouter's catalogue
price for `MODEL_NAME`, or custom rates from `CHAT_USD_PER_M_*`, see [docs/configuration.md](../docs/configuration.md)). `python -m evals.table evals/results` prints one row per eval and model over every
job folder. ALFWorld's game files sort by task type; `--stride 11` from offset 0 takes twelve games across
the six types. Every model plays the same tile draws because 2048 seeds the page's `Math.random`.

## Showcase runs and replays

The matrix runs headless and records nothing visual. Every episode still writes ``views`` into ``agent/episode.json``:
the state and the candidate keys before each decision, and the final state. That costs no extra call, because the
``act`` tool already reads them for its result.

``--showcase`` on any game writes the job under ``evals/showcase/`` (``evals.table`` never reads that tree) and, for
2048 and Millionaire, one PNG of the page per move into ``agent/frames/`` through the runtime, headless or headed.
``scripts/showcase.sh [SEED] [EVALS...]`` plays one episode per eval and model on the same seed, renders each pair
under ``evals/showcase/replays/<eval>/`` and copies the GIF to ``docs/results/<eval>/showcase/``.

``python -m evals.replay <trial> [<trial>] --out DIR [--gif]`` (the ``report`` extra: Pillow and Playwright) writes a
page with the two models side by side on the episode's own clock: the hands for Blackjack, the board for 2048, the
question card for Millionaire, the transcript for ALFWorld, and under each the chosen key, Jev's probabilities over
the candidates, the latency and the cumulative seconds. ``--gif`` screenshots the page per tick of episode time
(``--mode time --speed 4``) or per step (``--mode step``). ``--from-frames DIR [DIR]`` stitches one or two folders of
PNGs into a GIF, two of them side by side on the wall clock. ``evals/replay/cast.py`` fills such a folder for a
Flights run: it is a stdio proxy in front of the Playwright MCP (``PLAYWRIGHT_MCP_COMMAND=<python>``,
``PLAYWRIGHT_MCP_ARGS="-m evals.replay.cast --frames DIR -- npx -y @playwright/mcp@0.0.78"``) that asks the run's own
MCP session for a screenshot after every browser call and saves it as ``t<elapsed ms>-<n>-<tool>.png``, with the calls'
request and response times in ``calls.jsonl`` beside them. The policy behaves as it does without frames, since no
second browser client is involved. The frames hold the page and nothing else. ``python -m evals.replay <jev logs>
<llm logs> --out DIR --gif --strip`` takes two runs' logs folders (``answer.json`` inside, the frames under ``frames/``)
as trials: the page draws the frame at the clock with the tick's target probabilities under it, and ``--strip`` writes
the frames side by side under a header band. ``scripts/browser_showcase.sh <agent> [RUNS]`` runs both models headed
with the frames on and renders both GIFs from the median run of each model.

ALFWorld's scenes render through AI2-THOR: ``evals/replay/thor_replay.py <trial>`` replays the trial's commands in
``AlfredThorEnv`` and writes a frame per step; the page shows it above the transcript. It needs the ``alfworld-visual``
extra (``uv sync --extra alfworld-visual``): ai2thor 2.1.0, the Unity build the games were generated in, which ai2thor
downloads under ``~/.ai2thor`` on the first run (400 MB, x86_64, Rosetta on Apple silicon). ``scripts/showcase.sh`` runs
it for both ALFWorld trials before the page; without the extra the GIF holds the transcript alone.

## Status

Smoke-tested through the agent, one to three episodes each, Jev on the direct TypeSafe backend: Blackjack (rule and
jev), 2048 (jev with rethink on: 42 acts, 6 repeat blocks, 1.1 s per act, Jev median 382 ms), Millionaire (jev: 32,000
after 12 answers), ALFWorld (both won; the oracle plan in 12 steps, jev in 5).
Series of 2026-09-19 (Blackjack N=100 per model, ALFWorld N=12 per model, 2048 N=5) are tabulated in
`docs/benchmarks.md`; the 500-hand Blackjack series the protocol asks for has not been run.

Known gaps: the semantic-stall plan has not fired in a live run yet (2048 kept scoring). On Google Flights--1
Jev paged the date picker back and forth and ended BLOCKED after 18 requests (the date box, open issue 1 in `docs/browser-front.md`).

## Why the loop is shaped this way

1. Names: `evals`, `EvalState`, `create_eval_agent`, `ToolDecisionModel`; no "arena".
2. The anomaly rail is off for these agents. Its bailout ends a run on the third identical call, and `act(LEFT)`
   three times is a legal 2048 line. Exact repeats are `RethinkRail`'s first layer.
3. The rail and the slot model share one in-memory `EvalState` per episode. Session state was rejected.
4. Millionaire's 50:50 is a candidate key the model may pick when it is available, in place of a threshold wrapper.
5. `--model rule` keeps the hand-written baselines in the same loop.
6. The bare loop is gone; the loop-overhead number comes from `random` against the last bare-loop run
   (2048 random, 0.87 s per act).
7. Stall thresholds per game: 2048 six acts, ALFWorld eight, Blackjack and Millionaire none.
8. One slot model over one decision-model interface: `ToolDecisionModel` holds a decision model, and its `name` is
   the tick's `source` and the episode's `policy`.
