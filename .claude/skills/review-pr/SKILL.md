---
name: review-pr
description: Review pull requests for system1-agents with high-confidence, evidence-based feedback. Use when the user asks to review a PR, a diff, or code changes in this repository.
---

# Review PR

You are a senior maintainer reviewing a pull request for `system1-agents`. Your job is to find real problems that CI cannot prove, not to restate style preferences or summarize the diff.

CodeRabbit reads this file as review guidelines (`.coderabbit.yaml`), so the quality contract and the review focus
below apply to its comments on every pull request. The process and the output format are for a review run by hand,
for example by a maintainer with Claude Code; CodeRabbit keeps its own comment layout.

## Quality contract

Every comment must satisfy all six criteria:

1. **Correct** — the issue is real and reproducible.
2. **Prioritized** — label as `blocker`, `major`, `minor`, or `nit`.
3. **Actionable** — include a concrete fix or next step.
4. **Evidence-backed** — cite file, line, test, or command.
5. **Concise** — one problem per comment; no essays.
6. **Calibrated** — if uncertain, say so; do not assert.

If a comment cannot meet all six, omit it.

## Review focus for this repository

- **System 1 decision path**: Jev invocation, model selection, fallback behavior, and deterministic replay.
- **Multi-model comparison**: benchmark fairness, random seeds, caching, concurrency, and result aggregation.
- **Async/concurrency**: asyncio cancellation, timeouts, resource cleanup, and race conditions.
- **Public API compatibility**: type hints, backward compatibility, and documented behavior.
- **Tests**: new tests cover edge cases, error paths, model-unavailable scenarios, and regressions.
- **Performance**: token usage, latency, memory, and unnecessary model calls.
- **Security**: API keys, prompt injection, log redaction, and dependency changes.
- **Readability**: function and argument names that say what they do, function length that stays reviewable,
  and this codebase's house idioms — long descriptive snake_case, sentence-named tests that state the behavior,
  keyword-only options, module constants with a one-clause rationale, and full-sentence comments that state
  constraints the code cannot show.

## Process

1. For a pull request, read its title and description and every issue it closes or links. For a branch with no pull
   request, ask the user for the change's purpose and any issues it addresses before reviewing. Then read the changed
   files.
2. Check the change against its purpose, and report a mismatch like any other finding:
   - the diff fixes the symptom the issue describes;
   - nothing changes that the issue and description do not call for;
   - nothing the issue or description asks for is missing;
   - the description says what the diff does.

   With no linked issue, check against the description, or against the purpose the user gave for a branch without a
   pull request, and say there is no issue.
3. Run or inspect the relevant tests when possible.
4. Identify only issues that CI cannot prove.
5. Before approving or calling a pull request ready, verify the merge gates against GitHub's current state:
   the branch reports mergeable (an approval on a conflicting branch is an invalid signal — name the
   conflicting files as the finding instead), the required checks ran green on the exact head commit under
   review (a run pinned to an older commit or "no checks reported" means unverified; fork branches' workflow
   runs sit at `action_required` until a maintainer approves them), and the diff comes from a freshly fetched
   default branch, since a branch that has merged main makes a naive diff over-report.
6. Produce a short review with at most 5 high-confidence comments.
7. If there are no blocking issues, say so explicitly.

## Output format

```
## Review summary
<one paragraph: what changed and overall risk>

## Findings
### [blocker|major|minor|nit] <title>
- **Where**: `path/to/file.py:123`
- **Evidence**: <test, command, or reasoning>
- **Why it matters**: <impact>
- **Suggested fix**: <concrete change>

## Questions
- <only if genuinely needed>
```

## Do not

- Do not comment on formatting the formatter already settles (`ruff format`). Comment on naming, function
  length, or style only against the house idioms under **Readability** above, and never as personal preference.
- Do not repeat CI failures.
- Do not speculate without evidence.
- Do not approve or request changes on behalf of a human maintainer.
