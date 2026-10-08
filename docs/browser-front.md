# The browser front: the policy, decision by decision

Code: `s1a/browser/` (`decision_model.py`,
`action_space.py`, `probe_js.py`, `prompts.py`), the decision-model layer in `s1a/decision_models/`
(`docs/decision-models.md`) and the HTTP transport in `s1a/decision_models/wire.py`. Tests: `tests/test_browser_policy.py`. The
harness side (the `DecisionPolicyModel` Protocol, `probe_for_policy` and `activate_page` on the Playwright
runtime, the policy path in `create_browser_agent`) is the decision-policy slot pinned in `pyproject.toml`.
Measurements: `docs/benchmarks.md`.

## Background

Every step of openJiuwen's browser subagent is one chat-model turn: the model reads a 12k-character
`<browser_state>` and writes a tool call as free text. Measured at seconds per step, with targets the runtime
sometimes has not registered. Jev is a System 1 decision model: one request holds a `state` and several
`choice` questions, the answer holds one option per question with a probability distribution and a
confidence, and the option can only be one of the offered indices. The reference implementation is
browser-use/jev-ultrafast (MIT), whose observe-decide-act tick this policy follows.

## Decisions

1. A decision model fills the `Model` slot. `BrowserDecisionModel(Model)` answers a browser turn (the tool list holds
   `browser_click`) with exactly one `browser_*` tool call and forwards every other turn (summaries, typed
   values) to the wrapped chat model. DeepAgent, rails, checkpoints and the permission engine are untouched.
   The model behind it is a `DecisionModel`: TypeSafe Jev over HTTP with `--model jev`, Laya in
   process with `--model laya`; the policy is the same. `action_space.py` builds the `Observation` and the
   typed questions of a tick (`build_observation`, `build_questions`) and reads the answer back onto a
   candidate (`interpret`).
2. The policy has its own in-page probe. `probe_js.py` waits for the page to settle and describes every
   visible control once. Each control is stamped `data-opens1a=<id>`; the stamp is the element's
   identity across ticks and the unique selector the runtime registers. `browser_click(target_id)` then passes
   the runtime's target validation. The runtime's own snapshot was not reused: 85 to 156 ms per call,
   empty `<input>` values, a 40-item cap, and a selector-uniqueness heuristic that drops most of Google's
   controls.
3. Settle first, decide once. The probe returns after `readyState`, a 60 ms DOM-quiet window (cap 500 ms) and,
   after typing into a combobox, rendered autocomplete options (cap 900 ms, document-wide `[role=option]`).
   Settling uses timers; `requestAnimationFrame` never fires in a hidden tab. A last-resort resolver keeps an
   unresolved promise from reaching the driver timeout. This policy discards no answers. jev-ultrafast decides
   on every page change and discards stale answers, 8 to 11 per run in measurement.
4. Hidden tabs are activated. A hidden tab throttles timers to about 1 Hz and does not render dropdowns.
   When the probe reports `visibilityState == "hidden"`, the policy calls `activate_page(url)` once and
   probes again.
5. The chat model generates typed values in the background before the field is reached. A `TYPE_TEXT`
   string is generated from the goal, the field, the page text and the history. With `--prefetch on`
   (`BrowserPolicy.prefetch_values`, the default) that call starts in the background for every editable field
   as soon as a probe shows it, keyed by URL host and path, stamp id and label. The value survives page-key
   changes within one document and is cancelled on navigation. Measured: value wait 0 ms with prefetch on
   (9 to 15 calls per run), 2.0 to 2.4 s off (3 calls). `--goal-values on` (`BrowserPolicy.goal_value_cache`)
   adds a `text_value` choice head over values extracted from the goal; off by default.
6. The harness recognises a policy structurally. `openjiuwen/harness/schema/decision_policy.py` defines a
   runtime-checkable `DecisionPolicyModel` Protocol with one method, `bind_runtime(runtime)`.
   `create_browser_agent` checks `isinstance(model, DecisionPolicyModel)`: no temperature copy, no
   LLM-only context processors, anomaly-detection rail off, `bind_runtime(browser_backend)`. The factory
   never imports this package.
7. One `_Run` per task. Goal, history, pending action, prefetched values, tick counters and the settle budget
   live in a dataclass; a new goal or a finished run starts a fresh one and cancels the previous run's
   background tasks. The decision model is `self._decision_model`; `Model.__init__` builds `self._client` as a
   real telemetry-bearing model client and inherited methods touch it.
8. Failures degrade to `BLOCKED`. A probe failure envelope (`{"ok": False, "error": ..., "elements": []}`)
   folds into a control-only action space (WAIT, DONE, BLOCKED); a decisions transport error, HTTP error,
   malformed 200 body or invalid distribution ends the turn with a `BLOCKED` summary that still holds URL,
   title, steps and page text. Response bodies never enter error messages or logs.
9. Answers are validated in the decision-model layer. `decide_many` accepts only a choice among the offered ids
   whose distribution covers exactly those ids, sums to 1 within 0.02, and peaks at the choice, for every
   head of the request, including the heads the choice did not select. An unusable answer is re-asked once
   with the same request (`DECISION_ATTEMPTS`); a transport error is final. `interpret` then reads a validated
   `Decision`.
10. WAIT is spent in-page. On a WAIT verdict the policy re-probes with a doubling settle window (500, 1000,
    1500 ms) until the `page_key` changes or the streak budget is gone; a decisions request on an unchanged
    page returns the same answer and costs a full request. Two bounds hold together: `MAX_CONSECUTIVE_WAITS`
    (5 verdicts) and `WAIT_SETTLE_BUDGET_MS` (3000 ms per streak, kept on the run so repeated verdicts cannot
    re-grant it). An exhausted budget is terminal.
11. Actions settle before the model is asked. When the probe after an action shows the same `page_key`,
    the policy waits in-page from 250 ms, doubling, up to `ACTION_SETTLE_BUDGET_MS` (1000 ms), counted
    against the same streak budget. Google Flights closes its date dialog 300 to 700 ms after "Done"; without
    this the model saw the old page and answered WAIT at request cost. Measured: 13 decisions and 2 waits
    per run became 12 and 1.
12. Two rules the rewrite had lost are back in `OPERATION_RULES`: a filled search field is not a submitted
    search, and a visible Search or Submit control with its required fields filled is pressed at once.
    Without them the model answered DONE before pressing Search in 3 of 3 runs.
13. Constants and their derivation (`decision_model.py`):

    | constant | value | constraint |
    |---|---|---|
    | `PROBE_SETTLE_MS` | 500 | equals the JS default; an un-escalated probe's timing is unchanged |
    | `MAX_PROBE_SETTLE_MS` | 1500 | `load(3 s) + settle + 1 s` last-resort must stay at least 1 s under the 30 s transport timeout |
    | `WAIT_SETTLE_BUDGET_MS` | 3000 | in-page wait one WAIT streak may spend before BLOCKED |
    | `ACTION_SETTLE_START_MS`, `ACTION_SETTLE_BUDGET_MS` | 250, 1000 | first and total post-action wait |
    | `DECISIONS_TIMEOUT_S` | 5 | Jev answers in 0.4 to 1.3 s through the proxy; a dead connection must not stall a step; one transport retry; `S1A_DECISION_TIMEOUT_S` overrides it for a slower System One server ([configuration.md](configuration.md)) |
    | `DECISION_ATTEMPTS` | 2 | one re-ask of the same request when an answer fails validation; a transport error is final |
    | `BATCH_ACTION_TIMEOUT_MS` | 2000 | a stamp lost to a re-render fails fast; the returned probe re-stamps |

14. Two decisions backends, separate keys. `typesafe` talks to `api.typesafe.ai/v1/systemone` with
    `TYPESAFE_API_KEY` and `jev-latest`; `openrouter` talks to `/api/alpha/decisions` with
    `OPENROUTER_API_KEY` and `typesafe/jev-1.13` (`TYPESAFE_API_URL` and `TYPESAFE_MODEL` override the
    proxy only). Setting `TYPESAFE_API_URL` selects the proxy backend, which reads `OPENROUTER_API_KEY`; the
    TypeSafe key is never sent to OpenRouter. Measured: 294 to 418 ms per decision direct, 450 to 510 ms through the proxy.
15. Batched actions (`--batch on`, `BrowserPolicy.batch_actions`, needs the `unsafe_dev` browser capability). Each step is one
    `browser_run_code_unsafe` call that performs the action and returns the next probe; one transport round
    trip per step. The target is the stamped selector from the last probe; a missing stamp re-probes
    and re-finds the element by role and label; an action that still fails returns the probe with `error`
    set, and the policy re-decides on fresh stamps. The runtime's target validation is skipped on this path.
16. DONE answers from the whole page. On DONE the policy probes once more with `all_text=True` (up to 20,000
    characters, viewport filter off) and asks the chat model for the answer; the terminal summary holds it
    under `answer`, next to a 2,000-character `page_text` excerpt for diagnosis. The harness caps the
    subagent's summary at 8,000 characters; the page text cannot be included in the final message.
17. PRESS_ENTER, a keyboard submit. An editable searchbox, textbox or combobox that already holds a value gets a
    `PRESS_ENTER` candidate next to its TYPE_TEXT and CLICK ones; an empty field, a read-only one and a button do
    not. Unbatched it is one `browser_press_key` of Enter, which goes to the focused field, the one just typed
    into; batched it is `target.press("Enter")` on the field's own locator. The case comes from the Allrecipes
    batch: the query was typed, then the site's own search button was clicked four times without a page change.
18. Dead targets are withheld. A control whose last two clicks left the page unchanged keeps its table row,
    marked `click_did_nothing`, and loses its CLICK candidate until a click on it moves the page again. The
    probe's occlusion check sees what covers a control now; a consent banner that returns after every
    navigation leaves the control looking clickable on each fresh probe. In the Allrecipes batch the run ended
    on the no-page-change guard after three such clicks.

## Rejected

- A `browser_jev_run` tool: the chat model would still decide when to delegate, one extra turn per step.
- Jev as a browser driver backend: a driver has eyes and hands only; the decision would be lost.
- Reusing the runtime's snapshot probe: decision 2.
- `requestAnimationFrame` for settling: never fires in a hidden tab.
- An "Open <label>" prefix on the click head for editable fields, as jev-ultrafast does: tried, 5 of 6 runs
  still chose TYPE_TEXT on the date box; withdrawn.
- Re-asking the model on an unchanged page after a WAIT: decision 10.

## Open issues

1. The date box: the model picks TYPE_TEXT on a field labelled "Departure" in 4 of 9 runs and the value
   model then types a city. Next step: put the input's `type`, `placeholder` and `autocomplete` into the
   element table so the operation question can see "this is a date field".
2. The declarative subagent path (`SubAgentConfig`) has no decision-policy option; only
   `create_browser_agent(model)` takes a policy.
3. The decisions client is a plain `httpx.AsyncClient`, outside the harness's connection pool.
4. Budget exhaustion is treated as terminal on the assumption that the model answers the same WAIT
   for an unchanged snapshot; if a live run shows otherwise, re-ask once before BLOCKED.
5. The runtime's `navigate`, `evaluate` and `press_key` are private; `s1a/tool/hands.py` reaches them
   through `_call_playwright_tool` and `_execute_probe_json` (upstream ask in `docs/roadmap.md`).
