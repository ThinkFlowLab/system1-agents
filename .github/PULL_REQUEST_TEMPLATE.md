## Required human contributor answers

Answer every question before implementation and PR submission, including drafts.
Reuse explicit, still-applicable human answers with a traceable source reference;
ask the human for anything missing and wait. An agent must not invent answers.
Blank fields, placeholders, or an agent-written rationale do not satisfy this gate.
Share only information and source links appropriate for this PR's audience.

1. **What outcome are you trying to achieve?**

   Answer:

2. **What is the highest-leverage variable for that outcome?**

   Answer:

3. **How will this work move that variable rather than create busywork?**

   Answer:

4. **Who is the customer or user, and what do they actually need?**

   Answer:

**After answering the four questions: is a PR needed at all to achieve this outcome?**

Human decision and rationale (or explicit source reference):

**For a bug fix: link actual observed failure evidence.**
Include the observed result of a minimal reproduction, failing regression test,
or relevant trace. Label untested diagnoses as unverified. For a non-bugfix,
state why this is not applicable.

Evidence:

<!-- Title: [Feat], [Fix], [Docs] or [Chore], then the outcome in one line. -->

## Why

<!-- The symptom: what a user, a run or a reviewer hits today. Link the issue if there is one. -->

## How

<!-- The root cause, then the fix. Name the files a reviewer should open first. -->

## What

<!-- Agents, flags, commands, records and docs that changed. Anything a caller has to change on their side. -->

## Verification

<!-- What you ran and what it printed, in a form a reviewer can repeat. -->

- [ ] `uv run ruff format --check . && uv run ruff check . && uv run ty check`
- [ ] `uv run pytest -q` and `scripts/smoke.sh`
- [ ] `CHANGELOG.md` and the docs say what the code does now

## Demo / evidence

<!-- Follow CONTRIBUTING.md#agent-video-demos and .agents/skills/self-review/SKILL.md:
Important PRs require a video of application/task + System1-Agents decision-model agent
and System1-Omni inference in the same run, with input → actions → observed result.
Choose a relevant README use case; use PR #35's recording as the example. Identify the actual inference engine;
reproduction command/fixture and exact baseline/head commits; measured comparisons
and raw results; useful demo/figure links with source revision, replay speed, and
limitations. Check asset rights and redaction. Label illustrations separately from
actual runs. Use N/A only for the guide's minor-change exemption. A missing required
video is a review gap: name its blocker/follow-up and keep the PR draft until it is
supplied or a maintainer accepts the documented exception. -->

- Video link, or exemption/blocker with follow-up:
- Application/task → agent actions → independently checked result:
- System1-Agents / System1-Omni commits, checkpoint and worker/frontend provenance:

For an agent-assisted self-review, use the [self-review skill](https://github.com/ThinkFlowLab/system1-agents/blob/main/.agents/skills/self-review/SKILL.md).
For recording and upload instructions, use the [agent video guide](https://github.com/ThinkFlowLab/system1-agents/blob/main/CONTRIBUTING.md#agent-video-demos).
