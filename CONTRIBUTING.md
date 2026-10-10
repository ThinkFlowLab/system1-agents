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

CI runs these on two installs: the core (`uv sync --extra dev`, on Linux with Python 3.11 and 3.13 on every
run, and on Windows with Python 3.11 on pushes to `main` and the weekly schedule, where the game tests skip)
and the contributor install above (on Linux, where the ALFWorld tests skip and the GIF test renders in the
Chromium that `playwright install` fetches). `uv build` and `ty check` run in the core jobs only, since the
extras resolve the optional imports `ty` is told to ignore.
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
  says how to verify. Include a **Demo / evidence** section; use the [self-review skill](.agents/skills/self-review/SKILL.md)
  and the recording guide below.

Apply [committed artifact hygiene](.agents/skills/self-review/SKILL.md#committed-artifact-hygiene)
in every review, including quick prechecks. Keep maintained fixtures and replay inputs; preserve raw run
evidence in durable, reviewer-accessible PR/CI artifacts rather than committing redundant generated output.

### Large code changes

PRs with **more than 3,000 changed lines of authored code** need extra contributor
attention before requesting review. Count additions plus deletions against the
PR's merge base in source files, tests, and build or validation scripts. Report
this count separately from the total diff size; exclude documentation, generated
output, lockfiles, and static fixtures from the code count, while still reviewing
those files for relevance and correctness.

- Complete a full self-review of every affected component and its integration
  boundaries. A quick precheck alone is insufficient; keep the PR in draft until
  the contributor self-review is complete.
- Consider splitting independent features, refactors, and cleanup into focused
  PRs. If the change needs to stay together, explain why in the PR description
  and provide a component map and suggested review order.
- Include the code-line count and a validation summary for each affected area in
  the PR description: commands, results, and unverified behavior with reasons.
  Cover changed interfaces between components as well as individual components.

Size signals the need for closer review; it is not itself a correctness finding.
Choose checks based on the changed behavior and risk. Crossing this threshold
alone does not require GPU benchmarks or other expensive experiments.

## Add an agent use-case recipe

Use [recipes/README.md](recipes/README.md) and [the template](recipes/TEMPLATE.md) to document a complete task
with an existing agent: prerequisites, inference backend, run command, independent result check, demo and
troubleshooting. Keep fixtures small and reproducible; distinguish tested profiles from instructions awaiting
a real run. Add the recipe to the index. The [add-agent-recipe skill](.agents/skills/add-agent-recipe/SKILL.md)
guides this workflow; the builder skill also links a new agent to its recipe/evidence handoff.

## Agent video demos

Important PRs must provide a video demonstrating **application/task → System1-Agents decision-model agent →
System1-Omni inference → observable application result**. This applies to new agents/use cases, material
workflow changes, decision-model/serving integrations, action/rail/recovery changes, and claimed task-quality
or performance improvements. Include dependency changes when they materially affect inference or execution.

Show all three parts of the same run:

| Part | What the video and linked trace establish |
|---|---|
| Application/task | Starting input/state, actual actions and an independent completion check or failure |
| System1-Agents | Agent invocation, real decision-model choices and their execution; identify chat-model work separately |
| System1-Omni | Ready worker/frontend, answering model/checkpoint and backend identity, and requests from this agent run |

The application can be a browser, desktop app, game, simulator or a text task such as ticket routing. A terminal
recording is valid when it shows the task, agent decisions, inference provenance and result. Aim for about
20–45 seconds; link the full trace and label cuts or replay speed. A demo illustrates one run, not a success
rate or speedup. Logs, screenshots, scripted controls and in-process/cloud-only runs support the evidence but
do not replace the required application + agents + Omni video.

Use [PR #35's ticket-router video and caption](https://github.com/ThinkFlowLab/system1-agents/pull/35#issuecomment-6017425034)
as the worked example: ticket inputs, `--model laya-served` through `omni-jev`, chosen routes and recorded
checkpoint/device identity. It uses Laya in the agent's decision-model slot; follow the same pattern for
other decision models with a verified System1-Omni serving path.

Minor documentation, formatting and test-only changes that do not alter behavior or introduce quality/performance
claims may use `N/A` with a concrete reason. A nonvisual workflow alone is not an exemption. If a required video
is blocked by model/modality/hardware support, a missing client, resources or disclosure rights, name the blocker,
link the available evidence and a follow-up, and keep the important PR draft until the demo is supplied or a
maintainer explicitly accepts the documented exception. Missing evidence must remain a review gap.

### Application candidates

Choose a scenario relevant to the change from the [README's use cases](README.md#what-ships) and
[agent guide](docs/agents.md). PR #35 is the recording/provenance example; the application need not be ticket routing.

| Application | Existing agents | What to show and check |
|---|---|---|
| Computer use | `desktop` | Real app/fixture window, offered controls, model-selected clicks and resulting display/state |
| Browser use | `flights`, `allrecipes` | Task goal, page controls, agent navigation/form actions and final results checked against the goal |
| Robotics / embodied task | `alfworld` | Task and observed environment, selected actions and environment-verified completion; identify text or visual replay |
| Games | `blackjack`, `game2048`, `millionaire` | Initial hand/board/question, actual moves and final game state/score |
| Support routing | `ticket_router` | Ticket input, model-selected queue, executed route and evaluator verdict |
| Guardrails | `injection_guard` | Tool-output input, guard decision and actual quarantine/pass behavior when attached to an agent |

Use the agent's supported platform/environment and a supported client/worker combination. A guard's labelled-set
evaluation can illustrate classifications, but demonstrating runtime quarantine needs the guard attached to an
agent. Keep fixture runs, simulated environments and recorded replays labelled. Every candidate still needs the
same System1-Agents invocation and System1-Omni inference provenance from that run.

### Prepare and record

1. Choose a small reproducible task or fixture and define the completion check: a saved file, a submitted form,
   a solved board, or recorded routes/verdicts. A model's `DONE` alone is not a completion check. Record the command,
   seed/inputs, budgets, source commit and any local modifications. Prepare dependencies, model downloads and
   warmup before recording; report them separately if discussing timing.
2. Identify the actual inference path: model/checkpoint revision, library or serving engine, device and settings.
   For a model already supported by [System1-Omni](https://github.com/ThinkFlowLab/system1-omni/blob/main/docs/supported-models.md),
   use its worker/frontend for the required demo. Check the client, modality and hardware support too. If the client,
   worker or required configuration is missing, state that blocker and link a follow-up under the policy above.
   Pin both repositories and retain endpoint/backend evidence; a `--model` name alone does not prove which engine
   answered. Keep chat-model calls identifiable. A changed checkpoint, precision or compilation setting is a
   separate configuration when comparing results.
3. Capture the agent's application and the relevant terminal output with a recorder you already use.
   [OBS](https://obsproject.com/kb/quick-start-guide) supports window/display capture on Windows and Linux and screen
   capture on macOS: add the capture source, check the preview, then start recording. On macOS,
   [Shift–Command–5](https://support.apple.com/en-us/102618) also records a selected screen portion. For a remote
   text agent, record the terminal showing its session. Use readable text and mute audio unless it helps explain
   the task; narration is optional.
4. Show the input or starting state, let the agent choose and execute its actions, then show the independent
   completion check or failure. For a fix, show the trigger and resulting behavior; include a baseline clip when
   it helps. Keep retries, failures and human intervention in the linked trace. Label cuts and playback speed,
   and give actual elapsed time if the clip compresses a longer run.
5. Watch the exported clip before uploading. Use safe sample data and check the frames, audio, captions, logs and
   metadata for secrets or private information. State asset ownership/attribution and whether the recording may
   be reused in project updates. Share only material you have permission to disclose.

For example, follow [the served-Laya setup](docs/served-laya.md#8-run-it) to prepare the System1-Omni worker and
Rust frontend. Once they are ready at ports 8000/8080, record the ready endpoint and a three-ticket agent run:

```bash
curl --fail http://127.0.0.1:8080/health
LAYA_SERVED_URL=http://127.0.0.1:8080 uv run s1a run ticket_router --model laya-served --rethink off \
  --episodes 1 --seed 0 --batch-size 3 --max-steps 3 --showcase --log
```

Show the ticket inputs and predicted/expected routes from the trial's `agent/episode.json`, as well as the
command output. The printed `job_dir` locates the run under `evals/showcase/`; these demonstration runs stay
outside the benchmark matrix. Show the answering identity and frontend URL from the same run's ticks
(`model`, `served_by`, `url` and `request_id`), alongside the worker/frontend revision and configuration.

For existing game/browser recordings, [showcase runs and replays](evals/README.md#showcase-runs-and-replays)
describe the frame capture and `python -m evals.replay` tools. Label their output as a **recorded replay**, with
the original run/commit and playback speed. Check a showcase script before using it: it can start both Jev and
chat-model runs. An upstream model's demo or a scripted policy does not demonstrate the current learned-model
agent integration.

### Attach the demo to the PR

Export a readable MP4 with H.264 where possible; aim below 10 MB. GitHub's
[attachment guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)
lists supported formats and current limits. Drag the clip into the PR description's **Demo / evidence** section
or a PR comment, wait for the upload to finish, and save the resulting link. Uploading makes the file public for
this public repository. Use attachments for videos; keep sanitized reproduction commands and maintained
fixtures in the repository, and generated run records in a durable reviewer-accessible archive or PR/CI
evidence, following [artifact hygiene](.agents/skills/self-review/SKILL.md#committed-artifact-hygiene).
Verify the uploaded video plays and that
reviewers can open its linked trace.

Use a caption such as:

```text
Task and completion check: ...
Outcome: ... (including failures/intervention)
Application/fixture and System1-Agents commits; baseline if compared; local changes: ...
System1-Omni commit; worker/frontend URL; answering model/checkpoint revision; device/settings: ...
Command, input/seed and budgets: ...
Inference provenance from this run (request IDs/ticks/worker trace): ...
Recording: actual run or replay; speed/cuts; measured elapsed time if available: ...
Video and raw trace/results: ...
Reuse permission and attribution: ...
```

## AI review

CodeRabbit reviews each non-draft pull request when it opens and again on every push (Dependabot's excepted), and checks
the change against the issues it closes (`Closes #N`). It needs the CodeRabbit GitHub App installed on the repository by
an owner; reviews of public repositories are free. `.coderabbit.yaml` holds its settings and the per-path focus, and it
reads `.claude/skills/review-pr/SKILL.md` and this file as the review criteria, so the criteria live in one place.
CodeRabbit cannot run tests; a maintainer verifies every finding. To review by hand with the same criteria, ask Claude
Code to use the `review-pr` skill on a pull request or a branch.

## Extras

Everything outside `openjiuwen` is an extra. An agent whose extra is missing says so on stderr and exits 1.

| extra | installs | for |
|---|---|---|
| `blackjack` | rlcard | `s1a run blackjack` |
| `alfworld` | alfworld, textworld | `s1a run alfworld`; also `ALFWORLD_DATA` and Python 3.11, see `evals/README.md` |
| `alfworld-visual` | the `alfworld` extra, ai2thor 2.1.0, torch | `evals/replay/thor_replay.py`, the AI2-THOR scene behind an ALFWorld trial in the replay page; the 400 MB Unity build downloads on first use |
| `report` | pillow, playwright | `python -m evals.replay`, the showcase pages and GIFs; `--gif` also needs `uv run playwright install chromium` |
| `visual` | pillow | `s1a run sokoban --visual`: the board renderer; text-board Sokoban stays dependency-free |
| `laya` | laya (torch, transformers) | `--model laya` on every agent and on `decide` and `probe`: Laya in process, no Jev key; the checkpoint downloads into the Hugging Face cache (`HF_HOME`) on first use |
| `cua` | cua-s1 (torch), huggingface-hub | `--model cua` on tool and browser agents and on `decide` and `probe`: Cua-S1 Nano in process; the 3 MB checkpoint downloads into the Hugging Face cache (`HF_HOME`) on first use |
| `omnijev` | torch, torchvision, transformers, peft, accelerate, safetensors | `--model omnijev` on the browser agents: OmniJev in process, no Jev key; the model code is a clone of the OmniJev repository named by `OMNIJEV_REPO`, the checkpoint and base model by `OMNIJEV_CHECKPOINT` and `OMNIJEV_BASE` (`docs/configuration.md`) |
| `dev` | pytest, pytest-asyncio, ruff, ty, jsonschema, pyyaml, referencing | the test suite, `scripts/smoke.sh` and the lint and type checks; the last three check the served-Laya fixtures against `docs/api/` |
| `cua-four-b` | the `cua` extra, transformers 5, peft, torchvision, pillow | optional local Cua-S1 4B; set `CUA_S1_VARIANT=4b`, and `CUA_S1_MODALITY=text` or `multimodal` |

`uv sync --all-extras` installs all nine. The CLI runs from a checkout; a wheel install (`uv tool install`,
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
