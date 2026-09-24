# The decision-model layer: one interface between every front and the decision model

Code: `s1a/decision_models/`. Tests: `tests/test_decision_models_*.py`, the shared contract in
`tests/decision_model_contract.py`. The wire itself (the async HTTP client, the `typesafe` and `openrouter` backends)
stays in `s1a/decision_models/wire.py` and reads no answer.

Every front that asks "which one" (the tool loop, the browser policy, the rails, `decide`, `probe`, the MCP server)
talks only to `DecisionModel`; the wire client is private to `s1a/decision_models/`. A decision model is a classifier over options the caller enumerates. It
reads a state and returns a distribution over the offered keys. Hugging Face writes "System 1 decision model" and
TypeSafe "System One model". This repository uses the terms interchangeably. `--model` picks the backend: `jev` (TypeSafe Jev over HTTP),
`laya` (in process, behind `uv sync --extra laya`), `cua` (Cua-S1 Nano in process, behind `uv sync --extra cua`),
`random` and `rule` (the tool front's baselines).
`build_model(model_name, seed=, rule=)` builds one from the environment; `llm` names the chat model, which `build_model` does not build.

## The interface

Input: an `Observation(state, images=())` (a JSON object or plain text) and typed questions by name:
`ChoiceQuestion(options, goal=, rules=, operation=)` picks one key among `options` (a string per key, or a dict for
the browser's element rows); `NoulQuestion(question, criteria=)` asks whether a statement holds. Output: a
`Decision(answers, latency_ms, usage, model, raw)` with one `Choice(key, probabilities, confidence)` or
`Noul(p, confidence)` per question name.

```python
model = build_model("jev")
decision = await model.decide_many(
    Observation({"player_total": 18, "dealer_card": 9}),
    {"pick": ChoiceQuestion({"hit": "take a card", "stand": "keep the hand"}, rules="stand on 17 or more")},
    attempts=2,
)
decision.choice("pick").key, decision.latency_ms, decision.usage.input_tokens
```

`decide_many` is the one place validation happens. It refuses, with `MODEL_SERVICE_CONFIG_ERROR`, an empty
question set, an image on a text-only model and a question type the backend lacks. It checks every answer:
a choice must name an offered key, its probabilities must cover exactly the offered keys, lie in [0, 1], sum to 1
within 0.02 and peak at the key; a noul probability lies in [0, 1] and its confidence defaults to `max(p, 1 - p)`.
An unusable answer is re-asked with the same request up to `attempts` times (never on a deterministic backend, never
after a transport error) and then raises `MODEL_CALL_FAILED`. `decide`, `choose` and `ask` are one-question
shorthands; `warm()` and `close()` open and release the backend.

## Backends

| `--model` | class | `name` | notes |
|---|---|---|---|
| `jev` | `JevModel(transport)` | `jev` | the request body every front sent before the layer existed, byte for byte; `from_env` picks TypeSafe or the OpenRouter proxy |
| `laya` | `LayaModel(agent, model=)` | `laya` | one forward pass per call on a thread; `MODEL_SERVICE_CONFIG_ERROR` when `input_tokens` fills the window (Laya cuts the state silently; `LAYA_MAX_LEN`, `LAYA_HEAD_MAX_LEN` widen it); `ValueError` and `RuntimeError` from the library become `MODEL_CALL_FAILED` |
| `cua` | `CuaS1Model(scorer, collator, model=, context_bytes=, option_bytes=)` | `cua` | Cua-S1 Nano, one `score_elements` pass per request on a thread; choice questions only, text only, deterministic; the context is header, state and rules; the checkpoint reads its first 256 bytes, and the first overflowing request logs one warning; `from_env` reads `CUA_S1_CHECKPOINT`, `CUA_S1_SUBFOLDER`, `CUA_S1_DEVICE` |
| `random` | `RandomModel(seed)` | `random` | uniform over the offered keys, confidence 0, one seeded stream per episode; choice questions only |
| `rule` | `RuleModel(name, rule)` | the rule's name | one-hot, confidence 1; a key outside the menu raises `RuntimeError`, a bug in the rule |

`name` lands in every tick's `source` and in `Episode.policy`; the eval table's columns take their labels from it.

TypeSafe Jev answers `--model jev`. One request holds a `state` and one or more questions over options the caller
enumerates; the answer holds one option per question, a probability per option and a confidence, from one forward
pass, with no free text. Three heads: `choice` picks one key among the options, `noul` gives the probability that a
statement holds, `score` places the state on an ordered rubric. Input is capped at 32K tokens; the endpoint is
`api.typesafe.ai` with `TYPESAFE_API_KEY`, or OpenRouter's `typesafe/jev-1.13` on `/api/alpha/decisions`. Latency and
price: `benchmarks.md`. Laya (Convai Innovations, open weights, 0.4B parameters) and Cua-S1 Nano (Cua, 855K
parameters) answer the same `choice` question in process.

## What each front sends

- Tool front (`ToolDecisionModel`): one `pick` choice question over the candidates with the agent's rules; the state
  includes the plan and the harness notices when the rethink rail wrote them.
- Browser front (`BrowserDecisionModel`): `operation`, one `<op>_target` per head (CLICK, TYPE_TEXT, SELECT) and
  `text_value` when goal values are offered, all in one `decide_many(attempts=2)` over the page state; `interpret`
  reads the validated decision onto a candidate. Ticks keep `decision_ms` and `input_tokens` for the artifact readers.
- Rails (`DecisionModelRail`): one `check` question, noul or choice, banded by the spec's thresholds.
- `decide`, `probe` and the MCP server: one `pick` question; the printed dict stays `choice`, `probabilities`,
  `confidence`, `ms`.

Three oracle tests in `tests/test_decision_models_jev.py` pin the tool, rail and browser bodies to the pre-layer JSON.

## Doubles

`ScriptedTransport` fakes the wire under `JevModel` (the adapter's body building and payload reading run for
real; `bodies` records every request). `ScriptedModel` fakes the interface for front tests that need no wire.
`FakeLayaAgent` in `tests/test_decision_models_laya.py` stands in for the library. Nothing patches `httpx`.

## Adding a backend

1. `s1a/decision_models/<backend>.py` with `class <Backend>Model(DecisionModel)`: set `name`,
   `supports_images`, `question_types`, `deterministic`; implement `model` and `_decide`, which translates the
   questions and returns a `Reply` whose `answers` are the backend's own dicts; `decide_many` validates them into
   a `Decision`. Keep any heavy import inside `from_env()`.
2. A `case` in `factory.build_model` and the name in `DECISION_MODEL_NAMES`, `tool/loop.py::MODEL_NAMES`,
   `browser/browse.py::BROWSER_MODEL_NAMES`, `rails.RAIL_MODEL_NAMES` and `cli.DECIDE_MODEL_NAMES`.
3. `tests/test_decision_models_<backend>.py` with `Test<Backend>Contract(DecisionModelContract, IsolatedAsyncioTestCase)`
   plus the backend's mapping tests; a fake for its SDK lives in that file.
4. An optional extra in `pyproject.toml` and an env block in `.env.example` when it needs a dependency.

Laya is text only and reads a 512 to 1024 token window; it fits the tool front first. The browser front's element
tables, sent to Jev as-is, ran well past that window on a real page before a single instruction token was spent:
a JSON object per row, the full page text, and ten actions of history. The window check sums `input_tokens` over
the request's questions. On the browser front (two to four questions per tick) only a cut on every head raises
the config error above; a cut on one head goes unseen.

`laya_state` (`s1a/decision_models/laya.py`) folds a browser-shaped state before every call: `page.text` dropped
(the choice heads already carry each candidate's own text; the free-form dump is for the chat model's DONE
answer, which Laya never writes), each element row rendered as one short line instead of a JSON object, and the
last three actions kept instead of ten. On the WebVoyager-style pages measured while adding this, that is
roughly a tenfold reduction in the JSON-shaped state's size before the tokenizer sees it — the difference between
routinely filling a 512-token window and, on most pages, comfortably fitting it. It is on by default and skips
anything that is not the browser front's shape; `LAYA_COMPACT_BROWSER_STATE=0` turns it off. `--model laya` on a
page whose element table is still too wide for the window needs `LAYA_MAX_LEN` raised, same as before.
