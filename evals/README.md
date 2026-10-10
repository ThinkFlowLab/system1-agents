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
| Injection guard, public set (InjecAgent + AgentDojo) | precision and recall per source without the context-needed positives, balanced accuracy on the hard subset | `s1a run injection_guard --model jev --labelled-set evals/labelled/injection-public.jsonl`; numbers in `docs/benchmarks.md` |
| Bounded recovery on/off (3 local form fixtures) | completion on/off in a real headless Chromium, judged by the fixture server | `python -m evals.recovery --repeat 1` |
| Bounded recovery live (real decision/chat models, 3 local form fixtures) | completion on/off with the real models in a real headless Chromium, judged by the fixture server; cost null unless `CHAT_USD_PER_M_*` is set; optional `--validate-submission` variant | `python -m evals.recovery.live --model jev --repeat 1` |

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

The browser front and the desktop agent share the switch and the flags: `--rethink on|off`, `--rethink-attempts`
(default 3) and `--rethink-timeout` (default 15 s). On a browser or desktop stall the run spends one bounded attempt
that re-probes the page and asks the chat model for a plan, then the normal decision picks the action (browser:
`docs/browser-front.md` decision 19; desktop: the same `RecoveryLimits`). The attempts and the seconds across all
refreshes and plans are cumulative for the task and never reset by progress; `--max-steps` and `--timeout` still cap
it. Recovery is a loop mechanism, not a task result. `python -m evals.recovery` is a small repeatable on/off subset
for the browser front (issue 16): three local form fixtures (`normal`, `recoverable`, `permanently_blocked`) served
from a loopback server, each played in a real headless Chromium through the production `browse` front with
`create_browser_agent`, the registered `browser_*` tools, their generation ids and the same `RecoveryLimits`; batch
actions and `unsafe_dev` stay off. Success is the fixture server's recorded form POST, never the model's DONE. Each
trial opens its **own fresh fixture** (unique trial id, empty record namespace), so a submit that verified one trial
can never verify a later one; the arms alternate their run order across repeats to blunt startup-order bias. The
decision model and planner are clearly-marked **scripted doubles** (`evals/recovery/scripted.py`) - a controlled
fault that keeps typing until the production plan arrives - so this measures the **mechanism**, not a trained model
and not a real site. Scripted decisions cost 0 by construction; there is no real-backend mode (`--backend` is not an
option) because real-model token usage and pricing are not implemented here. It writes the usual Harbor job folders
under `evals/results/recovery/` and a paired `paired_summary.json`/`.md`. The paired completion counts every planned
trial, errors and timeouts included, and pairs recovery off/on per (task, repeat); it also reports the paired mean
on-minus-off deltas for model calls, elapsed seconds and wasted actions. All three recovery evaluators use
`model_calls = decision_calls + chat_calls`; `planner_calls` is the subset of chat calls used for replanning and is
never added again. Timing stays separate: `elapsed_s` is the whole trial wall clock (not inference),
`decisions_ms` is the scripted decision time measured with `perf_counter` (near-zero and not a benchmark), and
`recorded_probe_ms` sums recorded probe wall times (already including action settling). It excludes recovery,
final and unticked probes, so it is a diagnostic rather than total environment time.
`--repeat` must be >= 1 and arms and tasks must be valid; the CLI exits non-zero when a planned trial reports a harness error or timeout,
even if its form POST already verified, while an expected terminal
score such as `BLOCKED` does not fail the run. Native Windows desktop recovery still needs a supported desktop
machine.

Every episode records its ticks (key, confidence, probabilities, latency, tokens, whether a plan was in the
state) and its rethink events in `agent/episode.json`; the count of chat-model calls and their token sums go to
`result.json`. For `llm` the decisions are the chat calls that produced an `act`; a tick whose key was
not offered is kept with `accepted: false`, counted as an `invalid_key`, and skipped by the replay. Its iteration
cap is twice the game's budget, since the chat model spends turns on unknown keys.
A run that reaches the iteration cap has `result_type: error` in `extra`; its score still counts. An episode that
raised (a decisions failure, an env error) keeps `error` set: `summary.json` reports it under `errors`, computes
the score statistics over the `scored` episodes only, and `evals.table` skips trials whose `result.json` has
`exception_info`.


### Live real-model recovery subset

`python -m evals.recovery.live --model jev|laya|cua --repeat N --tasks normal,recoverable,permanently_blocked
--arms off,on --results-dir DIR` is the matching real-model run: the same fixtures, production `browse`,
`BrowserPolicy`, registered `browser_*` tools, generation ids and `RecoveryLimits`, but with `build_model` and
`chat_model_from_env` supplying the real decision and chat/planner models. Nothing is injected: no scripted
decisions, no standard plan and no hidden answer, so the run is the model's own. Optional flags: `--timeout`,
`--max-steps`, `--stall-after`, `--recovery-attempts`, `--recovery-timeout`, `--headed`. Success is still the fixture
server's recorded POST, never the model's `DONE`, and each trial opens its own fresh fixture and browser.

Counting is split and never conflated. Decision calls are counted by a thin proxy that appends its record at the
`_decide` entry and fills the wall time and outcome in a `finally`-safe step, so retries after an unusable answer,
raised calls and cancellations are counted for real (ticks are not a call count) and a failed call's tokens are
unknown, not zero. A successful reply with missing or malformed usage also stays unknown; an explicitly
reported zero remains a known zero. The proxy preserves the inner model's `name`, `question_types`, `deterministic`, `supports_images`,
`model`, `warm` and `close`. Planner attempts are counted from the production recovery events whose `stage ==
"planner"` (a planner call is also inside `chat_calls`, and the two are never summed). Chat call counts and tokens
come from the production `CountingModel` that `browse` wraps around the chat model; when a trial raises before that
summary exists they are taken from the outer `CountingModel`, which records the same real calls. A trial that raises
inside `browse` keeps its decision/chat counters and its fixture oracle: `verified` (the fixture's recorded POST) and
`errored` (a returned or raised run error) are independent, and only a trial that failed before the browser existed falls back to a
fabricated record. `cost_usd` is `null` by default, because the live eval never reads a provider catalogue; only an
explicit `CHAT_USD_PER_M_*` configuration values a trial, and that value is labelled an estimate, not a bill. Even
then a trial stays `null` when any chat or decision call did not report usage. (The production browser path's own
`usage_summary` may look up OpenRouter's catalogue; the live eval deliberately ignores it.) Model load time is
measured once and listed separately from the per-trial numbers; a setup failure (a model that will not build, a chat
model with no key) exits non-zero without inventing any trial, and the already-built decision model is still closed.

Each run writes a directory under `<results-dir>/recovery_live/<run-id>/` (the run id has a unique suffix) with
`manifest.json` (the planned manifest first, then the actual chat `model_config` name and provider, the relevant
`LAYA_*`/`CUA_S1_*` context variables, model ids, platform, config, load times and cost basis), `trials.json` (every
planned trial, errors and timeouts included), `summary.json` and `summary.md` (per-arm and per-task completion,
recovery events/terminations, decision/planner/chat attempt counts and timings, wasted actions, token-known status).
Each finished trial is also written on its own under `trials/`, so a batch scheduler's kill does not erase completed
evidence; there is no resume engine. `--repeat` must be >= 1 and the arms/tasks must be valid; the CLI exits
non-zero when a planned trial reports a run error or could not be set up, including a timeout after a verified
POST. The result is honest about its scope: three small
synthetic pages driven by a real model, not an open-task success rate, and recovery may not trigger at all.

#### Opt-in server-side validation (exploratory follow-up)

`--validate-submission` (also `LiveConfig(validate_submission=True)`, recorded in the manifest's `config`) is a small
optional variant for a **prospective exploratory follow-up**, not a change to the default fixtures or their reports.
When it is on, `run_live` uses validated equivalents of the supplied tasks: the fixture server answers a real POST
whose value differs from the task's expected value with HTTP 422, re-serving the same retryable form plus an ordinary
visible validation error (the submitted data and message are HTML-escaped), never a success result page. A correct
value still returns the result page and the independent oracle still verifies it. Every real POST, rejected ones
included, is recorded. Nothing is injected: the model still sees only the ordinary page and the goal, with no scripted
decision, plan or hidden answer. The original `DEFAULT_TASKS` and `LiveConfig()` keep `validate_submission=False` and
are immutable; `run_live` replaces only its own local copy, so the default behaviour is unchanged.

The variant exists because the first valid normal-form pilot submitted an empty value then terminated, which exercises
premature completion rather than a stall; the follow-up asks whether ordinary server-side validation can expose
naturally repeated ineffective actions for the bounded-recovery mechanism. Recovery improvement is **not assumed**. It
is exploratory and not an independent confirmatory benchmark: report its results **separately** from the original
unconstrained forms and never combine or average the two. Each keeps its own full denominator (every planned trial,
errors and timeouts included), its own natural trigger counts, actual planner/decision/chat calls, successes, latency,
wasted actions and termination reasons, even when there is no improvement.

### Native desktop recovery subset

`python -m evals.desktop_recovery` runs the matching normal / recoverable / permanently blocked cases on a
small Windows fixture. It compiles the committed C# source with the existing system .NET compiler and drives only
its own verified process through the real `CuaDriver`, `WindowEnv` and tool-agent loop. The decision model and
planner are scripted; the app's own result file is the independent verifier. No model key or GPU is required.
Windows, or WSL with Windows interop and an interactive Windows desktop, is needed; this does not extend the
production desktop CLI's Linux support.

```bash
python -m evals.desktop_recovery --driver /path/to/cua-driver.exe
S1A_DESKTOP_TESTS=1 S1A_DESKTOP_DRIVER=/path/to/cua-driver.exe \
  pytest -q tests/system/test_recovery_desktop.py
```

Each run gets a fresh directory, with a paired summary and Harbor job records. Every fixture is launched once;
a failed launch is recorded as an error. Errors/timeouts remain in the planned denominator
and cause a nonzero CLI exit; expected blocked tasks do not. The report records actual planner calls, accepted
actions whose observed progress did not change, recovery budgets, and the operator next action recorded by the
runtime in `episode.extra.terminal`, separately from the shortened display output. It does not invent an escalation
in the evaluator. Trial time includes fixture startup and the agent
episode, but excludes the shared compiler and driver startup.

The system test is skipped unless explicitly enabled. Driver binaries, generated fixture executables, environments
and result folders are local artifacts, not source contributions. See `tests/fixtures/desktop_recovery/S1AFixture.cs`
and `tests/system/test_recovery_desktop.py` for the fixture and the independent assertions.

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

Recovery run evidence is kept outside the maintained source tree. The
[served-Laya recordings and follow-up](https://github.com/ThinkFlowLab/system1-agents/pull/31#issuecomment-6014842354)
include both success and failure; the
[reproduction archive](https://github.com/prettygirlisnotme/system1-agents/releases/download/pr31-served-recovery-evidence-20261006/served-recovery-evidence-preview.zip)
contains all eight trial records, traces, pinned runtime reconstruction and recording helpers.
The earlier in-process experiment remains available as a
[frozen report](https://github.com/prettygirlisnotme/system1-agents/blob/bd54c1461a782c138e28801441ece47321b274bd/evals/recovery/RESULTS.md)
and [24 trial records](https://github.com/prettygirlisnotme/system1-agents/blob/bd54c1461a782c138e28801441ece47321b274bd/evals/recovery/laya-results.json).
These synthetic runs illustrate workflows and failure modes; they do not establish a general task-success improvement.
