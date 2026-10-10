# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semver.

## Unreleased

### Added

- Optional local Cua-S1 4B inference through the `cua-four-b` extra and `CUA_S1_VARIANT=4b`, supporting text and
  multimodal input. Desktop `--pixel-target` offers named screenshot points; clicks remain bound to the observed
  window and capture.
- Verified desktop text input through `--text`, `--text-target`, and `--text-mode` (insert or replace).
  Completion requires confirmed input and fresh field readback; `--verify-file` also requires matching contents
  in a freshly written file. `--window-title` selects an exact window and `--app-path` launches a macOS app bundle.
- Snake recipe and game client: `evals/snake` vendors the laya-mlx snake CLI
  (Apache-2.0) with single-game paced recording and a 16-game multigrid mode
  against any `/v1/systemone` backend; `recipes/snake` documents setup,
  verification, and recorded evidence (0 deaths; playback-speed-1 GIFs).

### Fixed

- `decide` rejects repeated `--option` keys before model setup instead of silently replacing an earlier choice
  description. The usage error names the duplicate key without printing either description.

- Laya browser-state compaction preserves the recovery plan for the next decision.
- Desktop recovery resets its detection window on observed progress without resetting its cumulative budget.
- Screenshot-enabled desktop recovery detects pixel changes while ignoring capture IDs and PNG metadata.
- Recovery errors omit provider exception bodies, and tool episodes retain the full structured terminal and
  permission-flow next step separately from shortened display output.
- Recovery evaluation preserves timeouts after verified submissions, escapes submitted result text, and keeps
  missing decision usage distinct from reported zero tokens. Paired reports share call-count conventions.
- Custom recovery tasks now reach each trial's fixture with their requested routes, page behavior and submission
  validation.
- Bounded recovery treats an empty, whitespace-only or otherwise blank planner answer (browser and desktop) as a
  planner failure: the run stops with the existing reason and next action, the one attempt and its active seconds
  stay charged and the fresh observation is kept, instead of recording a `planned` event with an empty plan.
- Fit-probe cases with no options, no accepted answer, or an accepted key outside the offered options now fail
  input validation instead of skewing the fit verdict.
- Windows development checks: the smoke script accepts CRLF output, shell scripts and Git hooks retain LF
  line endings, and tests check socket closure and invalid output directories without Unix-specific behavior.
  The core CI matrix now covers Windows with Python 3.11.
- Rail, tool, and browser evaluations charge Jev-rate input tokens only when the decision backend declares
  them billable. Local model token usage remains recorded without Jev API charges.
- Browser front: a WAIT whose in-page settle moved the page now records `page_changed: true` in the history, so
  the next state no longer shows that wait as unmeasured.

### Changed

- Core Windows CI (`core (windows, 3.11)`) runs on pushes to `main` and the weekly schedule, not on every
  pull request. Linux `core` and `full` still run on PRs.

### Added

- `--model clm`: CLM's `clm-serve` behind the decision-model interface, over its `/v1/systemone`. The engine owns
  everything after a frozen Qwen3-8B encoder, so no torch and no cloud key are needed on this side. Offered
  everywhere the other HTTP models are: `run`, `decide`, `probe`, rails, the browser front and the MCP server.
  See [docs/clm.md](docs/clm.md) and the [ticket-router evidence](evals/ticket_router/CLM.md).
- Bounded recovery on the browser front: `--rethink on|off` (default off), `--rethink-attempts` (3) and
  `--rethink-timeout` (15 s). A stall (no page change, an A-B-A-B loop, a repeated URL, a WAIT that moved nothing)
  re-probes the page read-only through the same runtime permission and asks the chat model for a plan under a
  per-task `RecoveryLimits` budget; the next normal decision still picks the action, the plan never executes and
  cannot widen the offered tools or add `unsafe_dev`. Only the detection windows reset after an attempt; attempts,
  seconds, history and ticks stay. `report()` and `decision_ticks.json` keep the recovery events, counts and
  termination, and a timed-out, failed or exhausted recovery is a clear `BLOCKED` even when a summary carries text.
  `--model llm` with `--rethink on` is rejected before the agent or browser is built.
- The desktop agent's bounded recovery takes the same `--rethink-attempts` and `--rethink-timeout` names and stalls
  after 3 actions without progress. Recovery failures include an operator next action; failed or cancelled chat
  calls remain counted, and incomplete token usage is reported as unknown cost.
- Small repeatable browser and native Windows recovery on/off fixtures, with independent completion checks,
  bounded failure cases and paired reports. These use scripted models to verify mechanisms, not model accuracy.
- `.coderabbit.yaml` and the `review-pr` skill: CodeRabbit reviews every non-draft pull request except Dependabot's, forks
  included, once an owner installs the CodeRabbit GitHub App, against `.claude/skills/review-pr/SKILL.md` and the linked issues.
  Maintainers can run the same skill by hand.
- Agent use-case recipe index, authoring template and contributor skill, with a runnable ticket-routing
  example and independent fixture verification.
- Contributor guidance for recording and attaching agent video demos, identifying the inference engine and
  checking supported System1-Omni paths, linked from the builder/self-review skills and PR template.
- `--model laya-served`: Laya served over HTTP by system1-omni's worker (or plain laya-serve), on every front that
  takes `laya`, in `decide` and in MCP `decide`. Configured by `LAYA_SERVED_URL` and optional `LAYA_SERVED_*`
  variables; no cloud key. Run records name the served checkpoint, revision and device. Design, API spec and the
  run steps: `docs/served-laya.md`, `docs/api/`.
- Tool-front ticks keep the answering model as `model`, and a served model's `served_by`, `url`, `request_id` and
  `server_timing`.
- `--model omnijev` on the browser agents: [OmniJev](https://github.com/tinnel123666888/OmniJev) (Apache-2.0), a
  Qwen3.5 vision-language decision model, in process behind `uv sync --extra omnijev` and a local clone named by
  `OMNIJEV_REPO`. It decides over a screenshot; `docs/decision-models.md`, `docs/configuration.md`.
- Browser front: a decision model that reads images (`supports_images`) gets a viewport PNG with every tick's
  observation, captured through the probe's run-code executor (a failed capture is tried once more; with no image
  OmniJev fails the call as `MODEL_CALL_FAILED`); the tick records its `screenshot_ms`. Jev, Laya and
  Cua-S1 read text only and see no change.
- `S1A_DECISION_TIMEOUT_S`: the deadline of one decision on the `jev` backend, 5 s when unset. A local System One
  server behind `TYPESAFE_API_URL` can be slower than Jev: on Google Flights, OneJev-27B on an A100 takes about 3.7 s
  a decision and more on the calendar page, so the 5 s deadline stopped every run at the eighth step; with 30 s it
  completed the task. `docs/configuration.md`.
- Every browser run saves the page the task ended on as `final.png` in `--logs-dir` before the browser is released,
  however the task ends, a timeout included. The answer and `answer.json` name the file in `screenshot`. A judge
  that grades the end state can read it, as Harbor's WebVoyager judge does at `/logs/agent/final.png`. A failed
  screenshot leaves `screenshot` null, records the exception type in `screenshot_error` and changes nothing else in
  the answer. A `final.png` an earlier run left in a reused `--logs-dir` is removed when the task starts.
- The MCP `decide` tool accepts `model="jev"|"laya"|"cua"`, defaulting to `jev`. Local backends use their
  optional extras and need no Jev API key.
- `docs/benchmarks.md`: the Google Flights driver comparison rerun on 2026-09-23 from Poland, every arm three times on
  both decision backends, next to the baseline rows in one table; the 24 S1A records, as one archive, and the chart
  under `docs/results/flights/rerun-2026-09-23/`.
- `laya_state` (`s1a/decision_models/laya.py`): folds a browser-front state to fit Laya's 512 to 1024 token
  window before every call — `page.text` dropped, one short line per element row instead of a JSON object, the
  last three actions instead of ten, a probe flag such as `"expanded": "false"` read as off, and "(no change)"
  only on an action measured as unchanged — roughly a tenfold reduction in the JSON-shaped state on the pages measured.
  On by default; `LAYA_COMPACT_BROWSER_STATE=0` turns it off. `docs/decision-models.md`.
- `laya_browser_question` (`s1a/decision_models/laya.py`): with a folded browser state, each browser question
  reaches Laya as the goal and the operation (the agent's long rules dropped) and each target option as its
  element's label and value. Laya fits a question's instruction and all its options into one `head_max_len`
  budget, so a 23-element target head left each option about six tokens, `12: {"element": "[`, and no
  element name. `text_value` gets its own ask; options that shorten alike keep their key; a blocked row keeps its
  overlay's name. Browser runs want `LAYA_MAX_LEN=1536` and `LAYA_HEAD_MAX_LEN=1024`: a calendar page's target
  head measures about 900 tokens.

### Changed

- Desktop click candidates omit disabled or unlabelled controls and application menu items, including in
  click-only tasks.
- Backend contribution guidance requires cold in-process model load timing, profiling above 5 s, and
  complete checkpoint coverage before skipping random weight initialization.
- `--model laya` loads in about 3 s instead of about 35 s: the encoder is built with transformers' weight init
  off, since the checkpoint replaces every weight. Weights and answers are unchanged.
- Important agent/inference PRs require an application + System1-Agents + System1-Omni video, following
  PR #35's worked example. Contributor skills, recipes and the PR template retain missing demos as review gaps.
- `--model laya` no longer draws the encoder's random weights before the checkpoint replaces them, which took most
  of a load of about 40 s on CPU. The `laya` extra now needs laya 0.3.9 or later, which skips the draw itself,
  and the lock moves from 0.3.5 to 0.3.20. Weights and answers are unchanged: laya 0.3.10 and later run a request
  of five or more questions in fp16 on MPS, which moves the answers, so `--model laya` keeps such requests in fp32
  unless `LAYA_MPS_AMP_MIN_ROWS` is set.
- `--model` picks the model on every agent, on `decide` and on `probe`: `jev`, `laya`, `cua`, `llm`, `random` or
  `rule`. The results table's column, the replay page's badge data and a browser run's `answer.json` name it
  `model` as well; the replay still reads the `slot` key of records written by 0.1.0.

## 0.1.0 - 2026-09-23

### Added

- Consolidated configuration reference table in docs/configuration.md detailing every environment variable, default value, and reader subsystem.
- Nine agents through the CLI and the MCP server: `allrecipes` and `flights` (browser use), `desktop` (computer use
  on Windows or macOS), `ticket_router` (30 labelled tickets to five queues), `blackjack`, `game2048`, `millionaire`,
  `alfworld` (games and embodied text), `injection_guard` (rail).
- Seven side-by-side replays under `docs/assets/demos/`, Jev against the chat model, and their table in the README.
  The browser one comes from `scripts/browser_showcase.sh`: a frame after every browser call through
  `evals/replay/cast.py`, then `python -m evals.replay` on the two logs folders, which reads a browser run
  (`answer.json`, the ticks or the chat calls, the frames) and draws the frame at the clock; `--strip` writes the
  frames under a header band.
- Every browser tick records the settled head's top probabilities and their labels, for the replay's bars; every
  browser run writes `answer.json` with its wall clock next to its records.

- Three decision models in the slot: TypeSafe Jev over HTTP, Laya and Cua-S1 Nano in process, plus the `llm`,
  `random` and `rule` comparison slots.
- `s1a decide` and `s1a probe` for one decision or a JSONL of decisions outside any agent.
- The caller skill for Claude Code, Codex, Cursor and Hermes, and the Claude Code plugin with the `s1a-browser`
  subagent.
- The builder skill `build-s1a-agent` under `.claude/skills/`.
- Showcase replays and GIFs per eval under `docs/results/`, and the Google Flights driver comparison in
  `docs/benchmarks.md`.
- Browser policy: a filled text-like field offers `PRESS_ENTER`, a keyboard submit for a site whose own
  search button does not submit; a control clicked twice without a page change is marked `click_did_nothing`
  and its click is withheld until a click on it works.
- Browser probe: accessible names drop escaped markup from page text (Allrecipes renders a literal `<img src=...>`
  in its recipe cards), an unclosed trailing tag included.
