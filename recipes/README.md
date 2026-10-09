# Agent use-case recipes

Recipes answer: **How do I run agent X with decision model Y for task Z, and check the result?**
Use the same setup → run → verify structure as [vLLM recipes](https://github.com/vllm-project/recipes),
with agent-specific evidence: enumerated actions, an independently checked task outcome, the actual inference
backend, and the required application + System1-Agents + System1-Omni video for important PRs, with a linked trace.

## Available recipes

| Use case | Agent / front | Execution profile | Validation |
|---|---|---|---|
| [Route support tickets](ticket-routing/README.md) | `ticket_router` / tool | Local CPU rule baseline; Laya/Jev alternatives, System1-Omni served Laya, or CLM over HTTP | Baseline smoke run; this fixture's required served-model video remains pending, and CLM measured 18/90 under this loop's framing ([evidence](../evals/ticket_router/CLM.md)) |
| [Play Snake](snake/README.md) | snake game client (`evals/snake`) / tool-style loop | System1-Omni native worker (H800); MPS dev path | 3×2400-step (0 deaths) and 16×600-step multigrid (0 deaths) recorded runs, playback-1 GIFs |

The recipe's validation section identifies what was actually run. A supported model or a passing unit test
does not establish that every agent/model/hardware combination was exercised.

## Add a recipe

Copy [TEMPLATE.md](TEMPLATE.md) to `recipes/<use-case>/README.md` and add a row above. Give the use case a short
hyphenated name. Keep small safe fixtures beside the recipe; link larger recordings and result archives.
Start from an existing agent. Adding a recipe does not require building another agent or changing its runtime.

Each recipe should have one complete path from prerequisites to an observable result. Keep alternative model,
engine and hardware profiles explicit, with their own validation status. Check both the agent client's contract
and [System1-Omni's worker support](https://github.com/ThinkFlowLab/system1-omni/blob/main/docs/supported-models.md)
before giving a served command. Record missing integrations as gaps.

Include the task/inputs, setup, exact run and verification commands, source/checkpoint revisions, supported
actions, a demo/trace, and troubleshooting. A recipe may document an untested profile, but it must say so.
Follow the [video guide](../CONTRIBUTING.md#agent-video-demos) and [self-review skill](../.agents/skills/self-review/SKILL.md)
for the required video, the PR #35 example and exemption/blocker handling. A partial recipe's rule/command
checks do not qualify its full application + agents + inference demonstration. A small demo is not a benchmark;
comparisons still follow the [evaluation protocol](../evals/README.md#protocol).

Use the [README application candidates](../CONTRIBUTING.md#application-candidates) when choosing the next recipe:
computer use, browser use, embodied tasks, games, support routing and guardrails. Each needs its own task outcome
and application + System1-Agents + System1-Omni evidence; the ticket-router example does not qualify the others.
