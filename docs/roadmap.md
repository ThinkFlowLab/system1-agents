# Roadmap: applications, benchmarks, model backends and System1-Omni

The next release should provide three reproducible application paths: ticket routing, browser forms/search,
and desktop document/save workflows. Each path needs a validated model/backend combination, an independently
checked outcome, a runnable recipe, benchmark records and a matching application + System1-Agents + System1-Omni demo.

Status checked on **2026-10-08**, against System1-Agents
[`5d89f85`](https://github.com/ThinkFlowLab/system1-agents/tree/5d89f85e702fc8dbdb7b719721e3cb0dda0bef4f)
and System1-Omni
[`4a79980`](https://github.com/ThinkFlowLab/system1-omni/tree/4a79980d8a75fd063cb3f8247e06215e288b18ac).
The live checklist is [roadmap tracker #54](https://github.com/ThinkFlowLab/system1-agents/issues/54).
Update that tracker as work lands and refresh this snapshot.
Checked boxes below describe the stated merged foundation; they do not mean every scenario/model/device
combination has been evaluated. An open PR remains pending until it is merged and its acceptance checks are met.

## Application scenario support

- [x] Ship the recipe index/template and application + agents + Omni demo requirements:
  [PR #43](https://github.com/ThinkFlowLab/system1-agents/pull/43),
  [PR #45](https://github.com/ThinkFlowLab/system1-agents/pull/45).
- [ ] Finish the reference ticket-routing recipe with independently checked routes, a served-model trace,
  and a video from that recipe's run. The served-Laya adapter and its demo are merged; the recipe still needs
  its own evidence. Related: [PR #35](https://github.com/ThinkFlowLab/system1-agents/pull/35),
  [PR #43](https://github.com/ThinkFlowLab/system1-agents/pull/43), [recipe](../recipes/ticket-routing/README.md).
- [ ] Add browser form submission, date selection and search fixtures with fixed starting states,
  reset procedures and independent completion checks:
  [issue #13](https://github.com/ThinkFlowLab/system1-agents/issues/13).
- [ ] Complete desktop typing, keyboard, scrolling and document/file-dialog workflows; verify the saved result
  and state which operating systems were actually tested:
  [issue #15](https://github.com/ThinkFlowLab/system1-agents/issues/15),
  [RFC #26](https://github.com/ThinkFlowLab/system1-agents/issues/26),
  [PR #27](https://github.com/ThinkFlowLab/system1-agents/pull/27).
- [ ] Complete a screenshot-dependent desktop task through Omni, starting with the alternating Save/Cancel
  fixture and its independently checked file changes:
  [issue #44](https://github.com/ThinkFlowLab/system1-agents/issues/44).
- [ ] Qualify bounded recovery on browser and desktop tasks: verified recovery or a clear stop within the
  attempt/time budget, with permissions retained:
  [issue #16](https://github.com/ThinkFlowLab/system1-agents/issues/16),
  [PR #31](https://github.com/ThinkFlowLab/system1-agents/pull/31).
- [ ] Save the final browser page as an inspection artifact alongside the independent task verdict:
  [issue #41](https://github.com/ThinkFlowLab/system1-agents/issues/41),
  [PR #42](https://github.com/ThinkFlowLab/system1-agents/pull/42).

## Benchmarks

- [ ] Integrate the public231 decision-service benchmark, including raw request/response evidence,
  accuracy, validity, calibration and per-round p50/p95 latency:
  [PR #52](https://github.com/ThinkFlowLab/system1-agents/pull/52).
- [ ] Establish the repeatable application baseline using the existing eval runner and record formats.
  Count failures, timeouts and unsupported tasks in completion denominators; report total time, decision
  latency, model cost and failure reason:
  [issue #13](https://github.com/ThinkFlowLab/system1-agents/issues/13).
- [ ] Compare recovery enabled/disabled on identical tasks and budgets; record completion, additional
  model calls, added time and wasted actions:
  [issue #16](https://github.com/ThinkFlowLab/system1-agents/issues/16),
  [PR #31](https://github.com/ThinkFlowLab/system1-agents/pull/31).
- [ ] Publish reproducible application comparisons with both source SHAs, checkpoint revisions,
  hardware/settings, all attempts, observed variability and raw artifact links. Separate preparation,
  readiness, inference, environment execution and total task time:
  [issue #13](https://github.com/ThinkFlowLab/system1-agents/issues/13),
  [PR #52](https://github.com/ThinkFlowLab/system1-agents/pull/52).

Decision quality and application completion answer different questions. Kernel/engine optimization and
its measurements belong in System1-Omni; application reports should link the exact engine configuration
and evidence. Before comparisons, declare the hypothesis, isolated variable, controls, success criterion
and run budget. A demo is evidence of one run, not a completion rate or speedup. See
[the eval protocol](../evals/README.md#protocol) and
[Omni's engine benchmark reports](https://github.com/ThinkFlowLab/system1-omni/tree/main/docs/benchmarks).

## Model backend support

- [x] Ship served Laya through the existing decision-model interface, CLI and MCP:
  [PR #35](https://github.com/ThinkFlowLab/system1-agents/pull/35).
- [x] Ship optional in-process Cua-S1 4B text/multimodal inference and screenshot target selection.
  This does not complete the Omni screenshot client:
  [PR #32](https://github.com/ThinkFlowLab/system1-agents/pull/32),
  [issue #44](https://github.com/ThinkFlowLab/system1-agents/issues/44).
- [ ] Publish an Agents capability matrix for model/checkpoint, input modality, question types, scenario,
  serving path and tested hardware. Track implementation and real validation separately; mark untested
  combinations explicitly. [Tracker #54](https://github.com/ThinkFlowLab/system1-agents/issues/54)
  owns the Agents matrix; reconcile it with
  [Omni's model tracker #83](https://github.com/ThinkFlowLab/system1-omni/issues/83) and
  [supported-models documentation](https://github.com/ThinkFlowLab/system1-omni/blob/main/docs/supported-models.md).
- [ ] Qualify Open-Jev 9B/27B on the text reference path and browser suite through the existing local
  System One endpoint configuration; record the actual served checkpoint rather than inferring it from
  the adapter name:
  [PR #36](https://github.com/ThinkFlowLab/system1-agents/pull/36),
  [PR #52](https://github.com/ThinkFlowLab/system1-agents/pull/52),
  [issue #13](https://github.com/ThinkFlowLab/system1-agents/issues/13).
- [x] Require cold-load timing for new in-process backends, separately from warm inference and episodes:
  [PR #51](https://github.com/ThinkFlowLab/system1-agents/pull/51),
  [issue #29](https://github.com/ThinkFlowLab/system1-agents/issues/29).

Start with Laya for ticket routing, evaluate Open-Jev for text/browser decisions, and Cua-S1 4B for
screenshots. Engine support alone does not establish fitness for a scenario. Client support for `choice`
and `noul` also does not establish `score` support merely because a worker offers it.

## Integration with System1-Omni

- [ ] Complete the desktop screenshot serving bridge: current capture/task/candidates to Omni Cua-S1 4B,
  validated answers back to the agent, capture/window binding through execution, and failure/stale-capture
  coverage. Keep image bytes out of serialized observation logs:
  [issue #44](https://github.com/ThinkFlowLab/system1-agents/issues/44),
  [issue #14](https://github.com/ThinkFlowLab/system1-agents/issues/14),
  [PR #32](https://github.com/ThinkFlowLab/system1-agents/pull/32).
- [ ] Agree and test the shared `/v1/systemone` contract, with model-specific capabilities and mappings,
  readiness, bounded deadlines, errors and response provenance. Reuse the existing decision-model seam:
  [Omni RFC #61](https://github.com/ThinkFlowLab/system1-omni/issues/61),
  [served-Laya design](served-laya.md).
- [ ] Qualify slow local-server deadline handling, including transport/server evidence and non-finite
  timeout rejection:
  [PR #36](https://github.com/ThinkFlowLab/system1-agents/pull/36),
  [PR #53](https://github.com/ThinkFlowLab/system1-agents/pull/53).
- [ ] Publish one reproducible setup/run/verify path and matching video/trace for each release application.
  Validate the first inference after readiness; identify both repository revisions, checkpoint, worker/frontend,
  device/settings and requests from the same run:
  [PR #35](https://github.com/ThinkFlowLab/system1-agents/pull/35),
  [issue #13](https://github.com/ThinkFlowLab/system1-agents/issues/13),
  [issue #44](https://github.com/ThinkFlowLab/system1-agents/issues/44),
  [PR #45](https://github.com/ThinkFlowLab/system1-agents/pull/45).

System1-Agents owns observations, offered actions, execution, recovery and application evaluation.
System1-Omni owns model loading, warmup, serving, accelerator execution and engine optimization.
Record chat-model work separately from decision-model inference.

## Delivery order and release gate

1. Publish the capability matrix and complete the ticket-routing text path; integrate the decision benchmark.
2. Establish the browser suite and use it to qualify model choices and bounded recovery.
3. Complete the desktop input/save path and the served screenshot bridge.
4. Publish recipes, benchmark reports, capability status and matching demos for the three application paths.

The release gate is **three reproducible application paths**, each with a validated model/backend
combination and an independently checked outcome. Required evidence gaps remain open even when an
implementation PR has merged. Follow the [recipe guide](../recipes/README.md) and
[video requirements](../CONTRIBUTING.md#agent-video-demos).

## Follow-up model and scenario work

These extend coverage after the reference paths; they are not additional gates for the next release.

- [ ] Integrate and evaluate CLM with a real encoder; distinguish the merged Omni stub-contract recipe
  from measured decision quality:
  [issue #24](https://github.com/ThinkFlowLab/system1-agents/issues/24),
  [PR #46](https://github.com/ThinkFlowLab/system1-agents/pull/46).
- [ ] Integrate and evaluate Julia-1 on a suitable labelled task:
  [issue #25](https://github.com/ThinkFlowLab/system1-agents/issues/25).
- [ ] Evaluate the proposed OmniJev vision browser adapter on the repeatable suite, and verify the
  actual inference path before claiming an Omni integration:
  [PR #33](https://github.com/ThinkFlowLab/system1-agents/pull/33).
- [ ] Complete the public injection-guard labelled set and runtime quarantine evidence:
  [issue #3](https://github.com/ThinkFlowLab/system1-agents/issues/3),
  [PR #23](https://github.com/ThinkFlowLab/system1-agents/pull/23),
  [PR #6](https://github.com/ThinkFlowLab/system1-agents/pull/6).
- [x] Ship the Snake recipe/game client as an additional application example:
  [PR #48](https://github.com/ThinkFlowLab/system1-agents/pull/48).
- [ ] Evaluate the draft Sokoban addition against the System 1 task-fit rule; deduction-heavy levels
  require a separate justification and measured baseline:
  [PR #34](https://github.com/ThinkFlowLab/system1-agents/pull/34).

## Longer-term design notes

The notes below retain earlier rail, branch-point and environment ideas. They are a backlog;
the next-release checklist and delivery order above take priority. Historical probe numbers and effort
estimates below are not new measurements or release commitments.

## Rails: one shipped, two to ship

### Prompt-injection guard: shipped, with a 20-item labelled set

jiuwen ingests untrusted pages through `browser_snapshot` and `fetch_webpage`. The `injection_guard` rail
(`s1a/agents/injection_guard.py`) ships: one rail on `after_tool_call` asks Jev "does this text instruct
the agent?" (`noul`) and quarantines the result; 20 of 20 on `evals/labelled/injection.jsonl` at a median of
464 ms. Still to build: the planted-page demo (the browser subagent visits a planted page and the guard flags the
planted instruction before the model reads it) and a labelled set from the public ones, InjecAgent (1,054 tool-output
injections) and AgentDojo (97 tasks, 629 security cases). jiuwen's own
`openjiuwen/harness/rails/security/prompt_security_rail.py` is pattern-based; `SafetyPromptRail` only injects
guideline text before model calls.

### Tool-call guardrail inside the permission engine

Jev decides allow or ask for calls the static policy cannot classify, on every call. Plug-in:
`openjiuwen/harness/rails/security/tool_security_rail.py` (`before_tool_call`, which calls
`resolve_interrupt`). Demo: "clean up ./dist" with a planted `rm -rf ~/`. Thresholds to start from:
auto-allow above 0.7, ask in between, deny below 0.1. Probe: 7 of 7; safe calls scored p=0.73 to 0.94, unsafe
ones (home-directory wipe, force push, env exfiltration, CI-file backdoor) p=0.01 to 0.03. Effort: a day, plus a
100-call labelled set for a precision and recall table. Public set: R-Judge (569 records, GPT-4o F1 74.42%).

### Severity and escalate-to-human after every turn

A rail that scores every ReAct iteration and pages a human before the agent spends ten more turns on a
blocker or does something hazardous.

- Hook: `AgentCallbackEvent.AFTER_REACT_ITERATION` for the per-turn check, `ON_TOOL_EXCEPTION` and
  `AFTER_TOOL_CALL` for error-bearing results (`openjiuwen/core/single_agent/rail/base.py`).
- State sent to Jev, a few thousand tokens: task goal, last assistant message, trimmed tool results, error
  text, iteration and elapsed counters, the previous severity.
- Heads: `score` over [nominal, degraded, blocked, hazardous, critical]; `noul` "a human should look at this
  now"; optionally `choice` over {continue, retry, ask the user, pause and page}.
- Action: the interrupt path the permission engine uses for "ask", plus `ask_user`, plus a webhook.
- Alert-fatigue guard: escalate on p at or above 0.8 for two consecutive turns, or on hazardous and above
  once; cooldown per incident key; every escalation logged with the probabilities.
- Measure: label 300 turns from stored trajectories; precision and recall of escalations; time from first
  blocker to escalation; wasted turns after a blocker, before and after.
- Cost: a 3K-token state per turn is about $0.0001.

## Branch points in jiuwen where a rule decides today

Principle: keep regex for syntactic checks and put Jev on semantic branches.

| # | Place | Rule today | Jev head | Before/after metric |
|---|---|---|---|---|
| 1 | Injection on tool results | the `injection_guard` rail, shipped | `noul` | attack success rate and utility under attack on InjecAgent and AgentDojo |
| 2 | Tool-call gate | tier rules, `shell_ast`, path extraction, destructive patterns | `noul` | false-allow rate and ask-rate on 250 labelled calls; R-Judge |
| 3 | Loop detection, `harness/rails/model_anomaly_detection_rail.py` | identical tool set and args across rounds; repeated output suffix | `noul` "is the agent making progress?" | F1 on 200 labelled windows; wasted tool calls per episode |
| 4 | Retryability, `harness/rails/tool_call_resilience_rail.py` | exception type plus marker substrings | `choice` retry, backoff, repair args, give up | accuracy on 300 labelled errors |
| 5 | Context offloading, `core/context_engine/processor/offloader/tool_result_budget_processor.py` | size thresholds | `score` "needed again?" | recall calls per episode; task success at a fixed token budget |
| 6 | Tool discovery, `harness/tools/tool_discovery/bm25.py` | BM25 with exact-name boost | rerank of the top 10 | top-1 and top-3 accuracy on 200 query-to-tool pairs |
| 7 | Browser progress, `harness/tools/browser_move/.../semantic_state.py` | digest of URL, filters and result counts | `noul` "did this action move the goal forward?" | accuracy on 200 action pairs; steps to completion |
| 8 | Stop conditions, `harness/schema/stop_condition.py`; `harness/goal/evaluation.py` | iteration, token and time thresholds; chat-model goal judge | `noul` "goal met?" per iteration | precision of stops; wasted iterations after the goal was met |
| 9 | Already-LLM decisions: intent detection, chat reranker, subagent dispatch, RL judge | chat model | `choice` or `score` | agreement with the LLM decision, latency, cost |

Measurement recipe, the same for every row: pull decision instances from the trajectory store
(`openjiuwen/agent_evolving/trajectory`), label 200 to 300 with an LLM plus a human spot check, run the
current rule and Jev on the identical instances, report accuracy or F1 and AUROC from Jev's probabilities,
then rerun a fixed task set with the switch on and report the downstream number. Publish accuracy only from
a labelled set of 100 or more per head.

## System 1 tasks where the reaction must be fast

Inside agents: interrupt handling (steer, abort or continue while the user types mid-run), stream gating
before a token reaches the user, turn-taking in voice agents, popup and CAPTCHA-presence handling in
browser and mobile agents, severity and paging, semantic-cache validity, dedup and anomaly gating on log
streams. Around agents: transaction-time fraud gating, autocomplete acceptance, support-ticket triage, game
and simulation reflexes. Deliberate work stays with the chat model: planning, arithmetic, constraint solving,
multi-step debugging.

## Benchmarks still to build

Selection rule: the environment enumerates the actions each step, the right pick is readable from a text
state, the chain is long, a scored baseline exists, no deduction is required. Built and running through the
eval agent: Blackjack, 2048, Millionaire, ALFWorld text, the desktop window and the ticket router (see
`evals/README.md`).

| Task | Action space | State source | Setup | Effort |
|---|---|---|---|---|
| ALFWorld, hybrid THOR render | admissible commands | text plus an AI2-THOR window | macOS build per ALFWorld README | 1 to 2 days |
| EmbodiedBench EB-ALFRED | numbered list, 171 to 298 actions | prompt text with `action id i: ...` | Linux, CUDA, X display; custom-model path posts to a `/process` endpoint | 3 to 5 days |
| MiniGrid, BabyAI | 7 actions | symbolic grid | pip install | 1 day |
| WikiGame | links on the page | link texts | none | 1 day |
| Minecraft via MINDcraft | 49 commands, typed params | query commands | Java server, Node 18 or 20; numeric params need buckets | 2 to 3 days |

EmbodiedBench and MINDcraft run their own loops and take an OpenAI-compatible endpoint; an
OpenAI-compatible server over `ToolDecisionModel` serves both. Not recommended for Jev: Minesweeper,
Wordle, Sudoku, real Sokoban levels, chess beyond one-move tactics, E-CommerceBench (numeric prices and
free-text negotiation decide the score).

## Later sequencing

After the next-release application paths and benchmark baseline:

1. The tool-call guardrail next to the shipped injection guard, as one "Jev security rails" feature, with the
   labelled sets.
2. Severity and escalation rail, measured on stored trajectories.
3. Branch-point rows 3 to 8 in order of expected delta.
4. The OpenAI-compatible server, then MINDcraft, then EB-ALFRED on a Linux GPU box.

## Upstream asks

- openJiuwen-ai/agent-core: the decision-policy slot (`DecisionPolicyModel` Protocol, `probe_for_policy`
  and `activate_page` on the Playwright runtime, the policy path in `create_browser_agent`), today on
  [ThinkFlowLab/agent-core#1](https://github.com/ThinkFlowLab/agent-core/pull/1), merged into that fork's `jiuwen-jev` branch; plus public `navigate`, `evaluate` and
  `press_key` on `BrowserAgentRuntime`, which `s1a/tool/hands.py` reaches through private methods today; and, for
  the final page `s1a/browser/browse.py` saves, a public screenshot on the runtime and a public handle to the runtime
  from the browser agent.
- microsoft/playwright-mcp: a switch for the two 500 ms settle sleeps in `waitForCompletion`.
