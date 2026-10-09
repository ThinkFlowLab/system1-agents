---
name: add-agent-recipe
description: Add or update runnable use-case recipes for existing System1-Agents agents, with setup, inference-backend configuration, independent result checks, demo evidence and troubleshooting. Use for recipe authoring; use build-s1a-agent when the task needs a new agent implementation.
---

# Add an agent use-case recipe

Read the target checkout's local instructions, [CONTRIBUTING.md](../../../CONTRIBUTING.md),
[recipe index](../../../recipes/README.md) and [template](../../../recipes/TEMPLATE.md).
Recipes live at `recipes/<use-case>/README.md`, with small safe fixtures beside them.

Resolve the concrete task, existing agent/front and observable completion check. Trace its state source,
enumerated choices, execution tools and evaluator; do not assume a `DONE` signal establishes success.
If it needs a new runtime behavior or backend integration, identify that separate scope before presenting
a runnable recipe for it.

Check the actual CLI, extras, environment precedence and artifact paths. State the decision model's
checkpoint/revision and library or serving engine separately from the chat model's role. For System1-Omni,
verify both worker support and the agent client's modality/question/window/hardware contract. Pin both
repositories and record a missing client or configuration instead of inventing a supported command.

Write one complete setup → run → verify path, then clearly identified alternatives. Use a bounded task and
independent result check; include errors, incomplete work and incorrect actions in the check. Label key-free
rule/scripted controls separately from learned-model inference. State required credentials by variable name
and expected downloads or external calls. Keep demonstrations outside measured benchmark runs.

Validate links, fixture schema and example flags against the implementation. Exercise the recipe only within
the task's authorized resources and budgets; follow applicable GPU reservation and experiment rules. Record
source revisions/local modifications, fixture hashes, environment, exact commands and raw results. Label
command-only checks and untested profiles explicitly. One working demo is not a quality or performance claim.

Prepare the demo using [the video guide](../../../CONTRIBUTING.md#agent-video-demos), including captions,
trace links and actual engine provenance. For important PRs, require a video of the application/task,
System1-Agents decision-model agent and System1-Omni inference in the same run, as in the linked PR #35 example.
Choose the relevant computer/browser/embodied/game/routing/guardrail application from the guide's README candidates.
A rule control or in-process run does not fulfill that requirement. Record missing support/resources as a
review gap and keep the PR draft unless a maintainer accepts the documented exception.
Retain failures, interventions and replay speed/source. Add the recipe to the index and link reusable
shared documentation rather than copying it. Use the
[self-review skill](../self-review/SKILL.md) for the final contributor report. Recipe authoring does not itself
authorize live model calls, downloads or external publication.
