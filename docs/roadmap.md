# Roadmap: the rails, the branch points and the benchmarks still to build

This file holds the work not yet in the repository: the rails, the branch points where a
rule decides today, the benchmarks still to build, and the asks to upstream projects. What exists is described
in `README.md`, `docs/agents.md`, `docs/browser-front.md` (the browser policy) and `evals/README.md` (the eval loop).

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

## Order of work

1. The model matrix on the built evals, with cost and decision counts (in progress).
2. The tool-call guardrail next to the shipped injection guard, as one "Jev security rails" feature, with the
   labelled sets.
3. Severity and escalation rail, measured on stored trajectories.
4. Branch-point rows 3 to 8 in order of expected delta.
5. The OpenAI-compatible server, then MINDcraft, then EB-ALFRED on a Linux GPU box.

## Upstream asks

- openJiuwen-ai/agent-core: the decision-policy slot (`DecisionPolicyModel` Protocol, `probe_for_policy`
  and `activate_page` on the Playwright runtime, the policy path in `create_browser_agent`), today on
  [ThinkFlowLab/agent-core#1](https://github.com/ThinkFlowLab/agent-core/pull/1), merged into that fork's `jiuwen-jev` branch; plus public `navigate`, `evaluate` and
  `press_key` on `BrowserAgentRuntime`, which `s1a/tool/hands.py` reaches through private methods today.
- microsoft/playwright-mcp: a switch for the two 500 ms settle sleeps in `waitForCompletion`.
