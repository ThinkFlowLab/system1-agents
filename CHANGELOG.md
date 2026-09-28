# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semver.

## Unreleased

### Fixed

- Bounded recovery treats an empty, whitespace-only or otherwise blank planner answer (browser and desktop) as a
  planner failure: the run stops with the existing reason and next action, the one attempt and its active seconds
  stay charged and the fresh observation is kept, instead of recording a `planned` event with an empty plan.
- Windows development checks: the smoke script accepts CRLF output, shell scripts and Git hooks retain LF
  line endings, and tests check socket closure and invalid output directories without Unix-specific behavior.
  The core CI matrix now covers Windows with Python 3.11.

### Added

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
- The MCP `decide` tool accepts `model="jev"|"laya"|"cua"`, defaulting to `jev`. Local backends use their
  optional extras and need no Jev API key.
- `docs/benchmarks.md`: the Google Flights driver comparison rerun on 2026-09-23 from Poland, every arm three times on
  both decision backends, next to the baseline rows in one table; the 24 S1A records, as one archive, and the chart
  under `docs/results/flights/rerun-2026-09-23/`.

### Changed

- `--model laya` loads in about 3 s instead of about 35 s: the encoder is built with transformers' weight init
  off, since the checkpoint replaces every weight. Weights and answers are unchanged.
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
