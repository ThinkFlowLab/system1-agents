# Why system1-agents: a decision model in the model slot of a stock openJiuwen agent

Sections: the problem (1), where Jev fits (2), design principles (3), value (4), packaging as a subagent (5),
precedents (6). The code map is `docs/architecture.md`, the numbers are `docs/benchmarks.md`, the open work is
`docs/roadmap.md`.

## 1. Problem

Every step of a tool-using agent is one chat-model turn. The model reads the transcript, which holds every
earlier observation and tool result, and writes the next tool call as text. On openJiuwen's browser subagent
that means a 12k-character `<browser_state>` per step, seconds per step, cost that grows with the transcript,
and tool calls that sometimes name targets the runtime never registered. On a game or an embodied task the
same loop pays the same price per move.

Many of those steps are selections. The page lists its controls, the game lists its legal moves, the household
simulator lists its admissible commands. The options exist before the model is asked. The decision is which of
them. A System 1 decision model returns a probability per option in one forward pass; a text generator writes the
same choice out token by token.

The question this repository answers: when a decision model fills the model slot of a stock openJiuwen agent,
what happens to speed, cost and score, on the same loop, the same tools and the same rules as the chat model?

## 2. What Jev is, and where it fits

TypeSafe Jev (`jev-latest` at api.typesafe.ai, `typesafe/jev-1.13` on OpenRouter) is a decision model with
three heads, each returning an index into options the caller enumerates: `choice` (one option key, a
probability per option, a confidence), `noul` (the probability that a statement holds) and `score` (a position
in an ordered rubric). It emits no free text. Input is capped at 32K tokens. Price: $0.042 per million input
tokens, output free. Measured latency: 350 to 420 ms per request direct, 450 to 510 ms through OpenRouter.

Probe evidence (`jev_probe.py`, four rounds, kept in the agtai/agent-core fork): 45 of 48 single-step decisions
right. The three misses were Minesweeper (constraint
deduction), Wordle (letter-position constraints) and a severity `score` of 2.47 on a 0 to 3 rubric where the
label was 3.

The fit rule that came out of the probes and the E-CommerceBench study: a task fits when the environment
enumerates the actions at each step, the right pick is readable from a text state, the chain of steps is long,
a scored baseline exists, and no step needs deduction or arithmetic. Recognition, goal matching and world
knowledge fit. Deduction, arithmetic, search and free-text generation stay with the chat model. E-CommerceBench
is the worked counter-example. Its score depends on generated prices and negotiation text. A decision model produces
neither.

## 3. Philosophy

1. The decision model sits in the model slot. `BrowserDecisionModel` and `ToolDecisionModel` subclass openJiuwen's
   `Model` and decide through one decision-model interface (`s1a/decision_models/`). The agent, its tools, its rails,
   its checkpoints and its permission engine stay as they are. Jev, Laya, Cua-S1 and a chat model are interchangeable
   by configuration. (`docs/browser-front.md`, decision 1; `docs/decision-models.md`)
2. Hands, brain and driver are separate parts. openJiuwen is the agent (hands), Jev is the brain, the browser
   driver (browser-use sidecar or Playwright MCP) is the driver. A driver has eyes and hands only. The decision
   belongs above it. (The agtai/agent-core fork's F_04 note, "Rejected")
3. One loop, swap the brain. Every eval runs the same DeepAgent with the same two tools and the same rules
   text; `--model jev|llm|random|rule|laya|cua` changes only the model. Score, seconds, steps, decisions and dollars
   are compared on the same seeds. (`evals/README.md`)
4. System 1 for Jev, System 2 for the chat model. Fast recognition (which control, which move, is this text an
   instruction, is this call safe) goes to Jev. Planning, arithmetic, constraint solving, typed values and the
   final answer go to the chat model. Both run inside one agent. The split is per turn. (`docs/roadmap.md`,
   "System 1 tasks")
5. Regex for syntax, Jev for semantics. Where a rule decides today inside openJiuwen (loop detection,
   retryability, tool gating, stop conditions), the plan keeps the syntactic checks and puts Jev on the
   semantic branch, with a before-and-after metric per branch point. (`docs/roadmap.md`, branch-point table)
6. Settle first, decide once. The browser policy asks Jev once per settled page and executes every answer.
   jev-ultrafast asks on every page change and discards 8 to 11 answers per run. (`docs/browser-front.md`, decision 3)
7. Answers are validated and failures degrade. A choice must be among the offered ids with a distribution
   that covers exactly those ids, sums to one and peaks at the choice. A probe or endpoint failure ends the
   turn as BLOCKED with a usable summary. Response bodies never enter logs. (`docs/browser-front.md`, decisions 8
   and 9)
8. Calibrated probabilities are the product. Each answer comes with a distribution and a confidence. A rail
   can set thresholds on them: allow above 0.7, ask between, deny below 0.1. (`docs/roadmap.md`, "Rails to
   ship")
9. Measure against a fair reference on one clock. Timings are quoted only from runs where every arm used the
   same Chrome, the same decisions backend and the same chat model, back to back. Day-to-day drift is
   attributed by component before any claim is made. (`docs/benchmarks.md`)
10. Seams upstream, fronts here. openJiuwen receives generic seams (the `DecisionPolicyModel` Protocol and
    two runtime hooks, about 100 lines). Every Jev-specific front lives in this repository.
    (`docs/roadmap.md`, "Upstream asks")

## 4. Value

- For users of openJiuwen agents: a browser subagent that finishes a Google Flights search in 7 to 12 s
  depending on the day and the driver, at a decision cost of $0.042 per million tokens.
- For long-chain discrete tasks (games, embodied text environments, quizzes): a model whose per-step cost
  stays flat while the chat model's transcript grows.
- For agent safety: rails with calibrated thresholds on prompt injection, tool-call gating and escalation,
  each with a labelled set and a precision-and-recall table; the injection guard ships (`s1a/agents/injection_guard.py`),
  the other two are in `docs/roadmap.md`.
- For other agents (Claude Code, Codex, Hermes, jiuwenswarm, agent-core DeepAgents): a specialised subagent
  for page tasks with enumerable controls, reachable as a CLI plus a skill (section 6).
- For TypeSafe: a reference integration of Jev inside a full agent harness, with the eval rig that measures
  it against a chat model per task.

## 5. How a System 1 agent becomes a subagent of other agents

Shipped as the `s1a` CLI, `s1a-mcp` and the caller skill `skills/s1a/SKILL.md`.

The finding that shaped it: none of Claude Code, Codex or Hermes has a cross-vendor "call another agent"
verb. Their subagent systems (`.claude/agents/*.md`, `.codex/agents/*.toml`, Hermes `delegate_task`) spawn
the host's own model instances. An external agent is reachable in three ways only: a shell command the host
runs, a tool call that runs the agent's loop (MCP "agent as tool"), or A2A (Hermes has a client, Claude Code
and Codex have none). The vendors that ship browser and research agents into these hosts lead with a CLI plus
a SKILL.md and keep MCP as a secondary or hosted transport (section 6). OpenAI's own Codex-in-Claude-Code
plugin delegates through the local Codex CLI and app server wrapped in a subagent and skills. OpenAI removed
`codex mcp-server` on 2026-09-05.

The plan follows that norm.

1. CLI first. `s1a run flights --model jev --goal "<goal>"` runs the browser subagent with Jev as its model and
   prints the answer as one JSON object on stdout; the harness logs go under `runs/logs`.
   `s1a decide --state @file --option a=... --option b=... --rules "..."` exposes the `choice` primitive.
   `s1a run <agent> --model jev --episodes N` runs a registered eval. A process start costs about 1.3 s (import of
   the openjiuwen browser stack). That is small next to a browse task. For `decide` it is three times the
   decision itself.
2. One skill, three hosts. `skills/s1a/SKILL.md` in the agentskills.io layout states when to call S1A
   (a page task with enumerable controls and no arithmetic), the commands, the result shape, and "use curl
   for a plain fetch". Claude Code reads it as a skill, Codex from `.agents/skills`, Hermes from
   `~/.hermes/skills`; `npx skills add ThinkFlowLab/system1-agents` installs it.
3. Host packaging. A Claude Code plugin (`.claude-plugin/plugin.json`, the skill, `agents/s1a-browser.md`
   with tools `Bash` and `Read`), in the shape of `openai/codex-plugin-cc`; Codex reads the same skill
   through the `.agents/skills/` symlink and Hermes takes the directory into `~/.hermes/skills`. openJiuwen DeepAgents and jiuwenswarm take the same skill through
   `create_deep_agent(skills=...)` and `SubAgentConfig.skills`.
4. MCP second. `s1a/mcp_server.py`, about 100 lines over the same functions, for hosts that pull tools only, and for
   `decide` at volume, where a long-lived process avoids the 1.3 s start. The `mcp` package is already an
   openjiuwen dependency.
5. A2A as a switch. openJiuwen exposes any registered agent as an A2A service (`openjiuwen/extensions/a2a`,
   `RunnerConfig(enable_a2a=True, distributed_mode=True)`). That gives the browser agent an Agent Card for
   Hermes `a2a_agents`, agent-core `RemoteAgent(protocol=A2A)` and the A2X registry. Claude Code and Codex
   gain nothing from it. ACP (Zed's editor protocol) is editor-to-agent and out of scope. The agent_teams
   bridge agent is out of scope too: S1A is a jiuwen agent already and joins a team as a native member.

Why this order, from the precedents in section 9:

1. Context. CLI calls "avoid loading large tool schemas and verbose accessibility trees into the model
   context" (Microsoft Playwright). A skill costs about 100 tokens until invoked (Anthropic).
2. Permissions. `Bash(s1a *)` rules in Claude Code and the Codex sandbox gate the call. An MCP
   agent-as-tool needs its own approval path.
3. Progress. A CLI streams ticks to the host as they happen. An MCP call returns one result, and Claude Code
   moves the call to a background task after two minutes.
4. Install. One command per host: `npx skills add`, `claude plugin install`, plus a checkout for the command
   itself (`git clone`, `uv sync`); a wheel install is unsupported.

## 6. Precedents: how others ship an agent into Claude Code and Codex

Verified 2026-09-19.

1. browser-use: the `browser-use` CLI plus SKILL.md is the documented Claude Code path. The MCP server keeps
   low-level tools plus one agent-as-tool, `retry_with_browser_use_agent`.
   https://docs.browser-use.com/cloud/tutorials/integrations/claude-code,
   https://docs.browser-use.com/customize/mcp-server
2. Vercel agent-browser: `npx skills add vercel-labs/agent-browser`. The skill is a stub that runs
   `agent-browser skills get core` to fetch instructions matching the installed binary. `agent-browser mcp`
   is optional. https://github.com/vercel-labs/agent-browser
3. Microsoft Playwright: `playwright-cli install --skills` for coding agents, MCP for "specialized agentic
   loops". README: "CLI invocations are more token-efficient: they avoid loading large tool schemas and
   verbose accessibility trees into the model context". https://github.com/microsoft/playwright-cli,
   https://github.com/microsoft/playwright-mcp
4. Browserbase Stagehand: the stdio MCP server is archived in favour of a hosted endpoint plus a skills repo
   whose skills shell out to a `browse` CLI. https://github.com/browserbase/mcp-server-browserbase,
   https://github.com/browserbase/skills
5. Firecrawl: `firecrawl init -y` installs the CLI and skills into every detected editor (Claude Code, Codex,
   Cursor, Hermes and others). MCP with `firecrawl_agent` stays as a transport.
   https://github.com/firecrawl/cli, https://github.com/firecrawl/firecrawl-mcp-server
6. Exa, Perplexity, Skyvern: hosted MCP, each with one long-running agent tool (`agent_run`,
   `perplexity_research`, `skyvern_workflow_run`). Exa also ships a Claude Code plugin with skills.
   https://github.com/exa-labs/exa-mcp-server, https://github.com/ppl-ai/modelcontextprotocol,
   https://www.skyvern.com/docs/integrations/mcp
7. OpenAI codex-plugin-cc: a Claude Code plugin that "delegates through your local Codex CLI and Codex app
   server", with the `codex:codex-rescue` subagent and `/codex:rescue`, `/codex:status`, `/codex:result`,
   `/codex:cancel`. `codex mcp-server` was removed in PR #42993. https://github.com/openai/codex-plugin-cc,
   https://github.com/openai/codex/pull/42993
8. Claude Code as a delegate: `claude -p` headless with `--output-format stream-json`; `claude mcp serve`
   exposes Claude Code's tools only. https://code.claude.com/docs/en/headless,
   https://code.claude.com/docs/en/mcp
9. Hermes: consumes MCP and skills; serves A2A (`a2a_agents`, `a2a_call`) and ACP. Third parties reach it
   through skills (Firecrawl, browser-use) or MCP (Skyvern).
   https://hermes-agent.nousresearch.com/docs/user-guide/features/skills,
   https://hermes-agent.nousresearch.com/docs/user-guide/messaging/a2a
10. Anthropic guidance: "MCP connects Claude to data; Skills teach Claude what to do with that data"; skill
    metadata about 100 tokens; plugins bundle skills, agents, hooks and `.mcp.json`.
    https://claude.com/blog/skills-explained, https://code.claude.com/docs/en/plugins
11. Protocol vocabulary: OpenAI Agents SDK `agent.as_tool` and handoffs; Google ADK `RemoteA2aAgent`; A2A
    1.0 under the Linux Foundation, with IBM's ACP merged into it; Zed's ACP is client-to-agent only.
    https://openai.github.io/openai-agents-python/tools/, https://adk.dev/a2a/quickstart-consuming/,
    https://a2a-protocol.org/latest/specification/, https://agentclientprotocol.com/overview/introduction
