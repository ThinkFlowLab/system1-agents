# Contributing

## Setup

```bash
uv sync --extra dev --extra blackjack --extra report
uv run playwright install chromium
cp .env.example .env
git config core.hooksPath .githooks
```

`uv sync` alone installs the package and the browser agents; the extras are listed below. `alfworld` and
`alfworld-visual` are opt-in: they need `ALFWORLD_DATA`, and the visual one a 400 MB Unity build. Nothing in the
test suite needs a key or the network. `.env` matters only for real runs.

## Checks

```bash
uv run ruff format --check .   # drop --check to fix
uv run ruff check .
uv run ty check
uv run pytest -q
scripts/smoke.sh
```

On Windows, run `scripts/smoke.sh` and the Git hooks from Git Bash with the native Windows `uv` on `PATH`.
The shell scripts and hooks keep LF line endings even when Git's `core.autocrlf` is enabled.

CI runs these on two installs: the core (`uv sync --extra dev`, on Linux with Python 3.11 and 3.13 and on
Windows with Python 3.11, where the game tests skip) and the contributor install above (on Linux, where the
ALFWorld tests skip and the GIF test renders in the Chromium that `playwright install` fetches). `uv build`
and `ty check` run in the core job only, since the extras resolve the optional imports `ty` is told to ignore.
`scripts/smoke.sh` is the contract for the core install: `list`, `--help` for every agent, `decide` without a key
and the MCP listing must work with no extra installed. A game whose extra is missing must say which one on stderr.

`tests/system` drives a headless Chromium through `@playwright/mcp`, which needs Node. It runs without a key:

```bash
S1A_BROWSER_TESTS=1 uv run pytest -q tests/system
```

CI runs it on pushes to `main` and every Monday. The job installs the Chromium revision that the runtime's pinned
`@playwright/mcp` expects.

## Hooks

`git config core.hooksPath .githooks` (in the setup block above) enables two hooks. `pre-commit` runs
`ruff format --check` and `ruff check` on the staged Python files, then `ty check` on the package. `pre-push` runs `pytest` and `scripts/smoke.sh`.
`git push --no-verify` skips them for one push. CI runs the same checks. Dependabot opens one grouped pull request a
week for Python packages and one for the GitHub Actions.

## Changes

- Write the test first. A new agent gets a test next to the others under `tests/`; the `build-s1a-agent` skill under
  `.claude/skills/` scaffolds one from a template.
- An agent that needs a package outside the core lists it as an extra in `pyproject.toml` and in `EXTRAS` in
  `s1a/run.py`. The missing-package line reads the extra name from there.
- `uv lock` after any dependency change; CI checks the lock file.
- Pull request titles use `[Feat]`, `[Fix]`, `[Docs]` and so on. The description answers why, how and what, and
  says how to verify.

## Extras

Everything outside `openjiuwen` is an extra. An agent whose extra is missing says so on stderr and exits 1.

| extra | installs | for |
|---|---|---|
| `blackjack` | rlcard | `s1a run blackjack` |
| `alfworld` | alfworld, textworld | `s1a run alfworld`; also `ALFWORLD_DATA` and Python 3.11, see `evals/README.md` |
| `alfworld-visual` | the `alfworld` extra, ai2thor 2.1.0, torch | `evals/replay/thor_replay.py`, the AI2-THOR scene behind an ALFWorld trial in the replay page; the 400 MB Unity build downloads on first use |
| `report` | pillow, playwright | `python -m evals.replay`, the showcase pages and GIFs; `--gif` also needs `uv run playwright install chromium` |
| `laya` | laya (torch, transformers) | `--model laya` on every agent and on `decide` and `probe`: Laya in process, no Jev key; the checkpoint downloads into the Hugging Face cache (`HF_HOME`) on first use |
| `cua` | cua-s1 (torch), huggingface-hub | `--model cua` on tool and browser agents and on `decide` and `probe`: Cua-S1 Nano in process; the 3 MB checkpoint downloads into the Hugging Face cache (`HF_HOME`) on first use |
| `dev` | pytest, pytest-asyncio, ruff, ty | the test suite, `scripts/smoke.sh` and the lint and type checks |

`uv sync --all-extras` installs all seven. The CLI runs from a checkout; a wheel install (`uv tool install`,
`pip install`) is unsupported, because the data folders (`evals/2048`, `evals/millionaire`, `evals/labelled`) sit
next to the package; the command refuses to start outside a checkout with one line on stderr. Runs, logs and
results go under the checkout (`runs/`, `evals/results`), or under `S1A_HOME` when that variable names another
root. Millionaire fetches its question ladders from the Open Trivia Database on its first run.

The `desktop` agent runs on Windows or macOS and has no extra. It needs Cua Driver: `cua-driver` on `PATH`, or the path in
`CUA_DRIVER_BIN`; macOS also needs Accessibility and Screen Recording granted in System Settings.

```bash
/bin/bash -c "$(curl -fsSL https://cua.ai/driver/install.sh)"
```

## Keys

`.env` needs a key for Jev (`TYPESAFE_API_KEY` direct, or `OPENROUTER_API_KEY` for the proxy) and for the chat
model (`OPENAI_API_KEY` or `LLM_API_KEY`, `MODEL_NAME`; `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` stand in
for `LLM_*` when those are unset). Exported variables win over the file. `--model laya` and `--model cua` need no
decision key. A host agent (Claude Code, Codex, Cursor, Hermes) reaches the keys through its own environment or
the checkout's `.env`.

## The openjiuwen pin

`openjiuwen` is pinned to a tagged commit on the `jiuwen-jev` branch of
[ThinkFlowLab/agent-core](https://github.com/ThinkFlowLab/agent-core). On that fork `develop` mirrors
[openJiuwen-ai/agent-core](https://github.com/openJiuwen-ai/agent-core). `jiuwen-jev` is `develop` plus the
patches this repo needs, one PR each on the fork and later upstream: the decision-policy slot,
`BrowserInstanceConfig.launch_args`, the loop-aware LLM client cache, and three browser-runtime fixes from the
WebVoyager runs. A release pins a `jj-<version>` tag on that branch. A sync re-pushes `develop` from upstream,
rebases `jiuwen-jev` onto it and retags. The first `uv sync` resolves that pin over the network and takes minutes;
openjiuwen itself brings a large tree (transformers, pymilvus, matplotlib, sqlalchemy). The sync also clones a
transitive dependency, `intelli-router`, from gitcode.com (`uv.lock` names it); a network that blocks gitcode.com
fails the sync.

## Skill and plugin layout

The caller skill, `skills/s1a/SKILL.md`, follows the agentskills.io layout. Claude Code, Codex and Cursor take it
through `npx skills add ThinkFlowLab/system1-agents`; Codex also reads it through the `.agents/skills/` symlink; Hermes
takes the same directory into `~/.hermes/skills`. The Claude Code plugin (`.claude-plugin/`) adds the
`s1a-browser` subagent and the MCP server. Every host runs the command from a checkout of this repository.
