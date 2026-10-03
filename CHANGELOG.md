# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semver.

## Unreleased

### Fixed

- Windows development checks: the smoke script accepts CRLF output, shell scripts and Git hooks retain LF
  line endings, and tests check socket closure and invalid output directories without Unix-specific behavior.
  The core CI matrix now covers Windows with Python 3.11.
- Browser front: a WAIT whose in-page settle moved the page now records `page_changed: true` in the history, so
  the next state no longer shows that wait as unmeasured.

### Added

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
