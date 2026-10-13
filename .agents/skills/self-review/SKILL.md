---
name: self-review
description: Self-review System1-Agents changes before opening a PR or requesting review, with reproducible task outcomes, appropriate demos, and evidence-backed claims.
---

# System1-Agents self-review

Start with the [human contributor-intent gate](#human-contributor-intent-gate).

Read the target checkout's `CONTRIBUTING.md`, PR template, and applicable local
instructions. Record the actual target branch and base/head commits, and review
the full diff from their merge base plus relevant uncommitted changes. Distinguish
what is in the PR from local-only work; disclose if the base could not be refreshed.

Before requesting review and after every push, refresh the base against the
default branch and confirm on GitHub that the pull request reports mergeable and
that the required checks ran green on the exact head commit offered — record that
head SHA in the report. A conflicted branch, or a green run pinned to an older
commit, is a readiness gap: name it rather than reporting CI from a stale state.
Fork branches' workflow runs sit at `action_required` until a maintainer approves
them; report that as pending, not as absent or passing.

Apply the [large-code-change requirements](../../../CONTRIBUTING.md#large-code-changes):
report authored-code and total diff counts separately using the guide's counting
convention. Above 3,000 changed code lines, verify the contributor's full
self-review, split rationale, component map/review order, and validation across
affected components and interfaces before recommending readiness. A quick
precheck is insufficient; keep the PR draft until the contributor self-review is
complete. Report missing preparation as a readiness gap; size alone is not a
correctness finding or a reason to require expensive model/GPU runs.

Check correctness, focused scope, decision-model and front contracts, fallback
behavior, cancellation/timeouts, and resource cleanup as relevant. Verify tests
cover the changed behavior, including failure paths and a regression case for a
fix. Match docs and commands to the implementation. Run the applicable checks in
`CONTRIBUTING.md`; report exact commands, outcomes, and skipped checks with reasons.
Documentation-only changes need link/example/claim checks, not unrelated model runs.
For use-case recipes, check the [recipe template](../../../recipes/TEMPLATE.md): commands reach the named
agent/backend, result checks can detect failure, and each profile's validation status matches its evidence.

This skill prepares a local contributor report. It does not itself authorize
edits, commits, pushes, external posts, paid model calls, downloads, or changes to
review status. Use only separately authorized execution resources and budgets.

## Human contributor-intent gate

Complete this gate with the human contributor before implementation and before
submitting a PR (including a draft). At pre-submit self-review, verify the answers
still describe the actual diff; ask again only where the scope or evidence changed.
This is a contributor preparation step, not a prerequisite for an independent
read-only review of someone else's PR, and it does not change automated review
selection or authorize any external action.

Before submission, verify that the [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md)'s
four required human answers and PR-necessity decision are complete and traceable,
and that bug-fix failure evidence is present when applicable. Blank fields,
placeholders, and agent-invented answers block implementation/submission. This is
a contributor workflow constraint; the Markdown template itself is not server-side
enforcement. Independent read-only reviews remain outside this gate.

Use explicit human answers already present in the task, issue, or discussion,
keeping a traceable reference to their source. Ask only for missing or ambiguous
answers, then wait for the human; do not infer, invent, or answer on their behalf:

1. What outcome are you trying to achieve?
2. What is the highest-leverage variable for that outcome?
3. How will this work move that variable rather than create busywork?
4. Who is the customer or user, and what do they actually need?

After those answers are clear, ask the human: **Is a PR needed at all to achieve
this outcome?** Reuse an explicit, still-applicable human decision if already
recorded. If the answer is no or remains unclear, stop implementation/submission
and clarify the next step instead of manufacturing a change to justify a PR.

For a bug fix, also link actual observed failure evidence, such as a minimal
reproduction with its observed result, a failing regression test, or a relevant
trace. Separate observed behavior from an untested diagnosis; a hypothetical
failure is not verified evidence. If the evidence is missing, ask for it or obtain
it within the authorized scope before proceeding.

Keep source references in the working task and include only audience-appropriate,
authorized summaries or links in the PR. Do not copy private conversations,
customer data, or credentials into a public repository. Missing human answers
remain a blocker; an agent-written rationale or completed checklist cannot
substitute for them.

## Committed artifact hygiene

Apply this check in every review, including quick prechecks. Inspect added and
changed artifacts in the complete diff, including JSON/JSONL, CSV, logs, reports,
source/binary hash inventories, and generated media. Classify them by purpose and
actual consumer, rather than rejecting a file extension.

- Keep necessary configuration, request examples, maintained benchmark inputs,
  and small deterministic fixtures or reference oracles in the repository's
  intended locations. Identify the test, tool, or documented workflow that needs
  each retained artifact, such as labelled agent datasets or replay inputs.
- Flag one-off run summaries, response dumps, cache statistics, profiler output,
  agent process notes, and duplicate historical results that have no maintained
  source-tree role. A link from PR prose or documentation alone does not justify
  committing generated run output. Report concrete paths and consumers.
- Preserve raw measurements, failures, and provenance in a durable artifact
  archive or PR/CI evidence, and link the exact revision or run from the summary.
  Do not discard evidence to reduce the diff or hide it in a committed archive.
- When removing redundant output, check its callers, links, and reproduction
  commands. Keep replay inputs and expected responses intact; verify their
  hashes and rerun the affected replay or documentation checks.

## Task evidence

For changes to what an agent can accomplish, show a concrete task as **input →
actions → final result**. Define success using an observable outcome, not just a
`DONE` signal. Use a safe, reproducible scenario or fixture; include setup,
commands, inputs/seed, model and configuration, environment, and exact source
commits. Link the resulting logs or artifacts so a reviewer can trace the story.

When claiming improvement, compare baseline and head on the same tasks, inputs,
success criteria, and budgets. Report completion counts/denominators, elapsed
time, model calls and tool calls when measured. Include cost only if measured,
with the accounting scope and pricing basis; do not infer total cost from latency
or an incomplete token charge. Keep failures, retries, timeouts, and human
intervention in the results. Report repetitions and variation; one successful
demo is not a task-success rate. Mark missing metrics unmeasured, and remove or
qualify unsupported claims rather than manufacturing a comparison.

Use [CONTRIBUTING.md's video guide](../../../CONTRIBUTING.md#agent-video-demos)
to classify the PR and prepare, record and attach its demo. Important PRs require
a video of the application/task, System1-Agents decision-model agent and actual
System1-Omni inference in the same run. Check all three parts against the linked
trace; a terminal recording works for text agents and rails. Screenshots and logs
support the clip. Use PR #35's recording linked in the guide as the example and
choose a relevant README application candidate from the guide. Show the
relevant input, action sequence, and result, with failures or human intervention
visible. Label cuts, replay speed, and elapsed timing honestly; link a fuller
trace when a clip omits context. Do not stage screens or present a replay as a
live run. A replay must identify the source
run/commit, workload, and speed; a historical or upstream model demo is not proof
that the current agent integration works.

Choose figures that answer the review question: workflow screenshots, a short
action timeline, or task-success comparisons backed by the run records. There is
no asset quota. Only the guide's minor docs/formatting/test-only exemption permits
`N/A` with a reason; a nonvisual task still needs a terminal video when the PR is
important. If a required run cannot be made within the available authorization,
resources, or budget, report the gap and its impact; keep the important PR draft
until the video is supplied or a maintainer accepts
the documented exception. Do not turn a missing run into a pass or require a
production-scale demonstration.

## PR demo/evidence section

Prepare a **Demo / evidence** section for the PR containing what applies:

- Required application + agents + Omni video and trace, or the documented exemption/blocker.
- Task and observable result, or `N/A` for an exempt change with a concrete reason.
- Reproduction command/fixture, configuration, and baseline/head commits.
- Measured comparison and raw result links, including failures and limitations;
  distinguish personally run checks, author-reported results, and observed CI.
- Demo/figure links with captions stating the workload, source revision, and
  whether each item is an actual run, recorded replay, or explanatory illustration.

Make evidence reusable for accurate reviews and public updates without implying
permission to publish it elsewhere. Before attaching assets, check ownership,
license/attribution, and permission to share. Use safe sample data and redact
credentials, private URLs, personal/customer information, and sensitive screen or
log content. Verify redaction in the final exported files, captions, and metadata.
Keep useful measurement context after redaction. Clearly label diagrams, mockups,
and generated artwork as illustrations; never fabricate screens, results, or
performance claims. Link durable, reviewer-accessible artifacts rather than local
paths. If rights or safe disclosure are unresolved, omit the asset and state why.

Check System1-Omni's current model, modality and hardware support as described in
the video guide. Use a supported serving path for the required video when the
branch has a compatible client; otherwise record the missing integration or
configuration. Pin both repositories and verify the actual worker/frontend from run evidence. Keep
agent-task evidence separate from serving/kernel measurements. Do not add an
unrequested backend integration or run outside the authorized resources/budget
merely to produce a demo.

## Report

Lead with actionable findings and file/line references, then the reviewed scope,
commands/results, demo/evidence summary, and remaining gaps. Say when there are no
actionable findings without implying maintainer approval. Keep blocking gaps
visible and recommend a draft while they remain. Do not check the contributor's
boxes or publish on their behalf without separate authorization.
