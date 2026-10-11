# The decision-model layer: one interface between every front and the decision model

Code: `s1a/decision_models/`. Tests: `tests/test_decision_models_*.py`, the shared contract in
`tests/decision_model_contract.py`. The wire itself (the async HTTP client, the `typesafe` and `openrouter` backends)
stays in `s1a/decision_models/wire.py` and reads no answer.

Every front that asks "which one" (the tool loop, the browser policy, the rails, `decide`, `probe`, the MCP server)
talks only to `DecisionModel`; the wire client is private to `s1a/decision_models/`. A decision model is a classifier over options the caller enumerates. It
reads a state and returns a distribution over the offered keys. Hugging Face writes "System 1 decision model" and
TypeSafe "System One model". This repository uses the terms interchangeably. `--model` picks the backend: `jev` (TypeSafe Jev over HTTP),
`clm` (CLM's `clm-serve` over HTTP, see [clm.md](clm.md)),
`laya` (in process, behind `uv sync --extra laya`), `laya-served` (Laya served over HTTP by system1-omni, see
[served-laya.md](served-laya.md)), `cua` (Cua-S1 Nano in process, behind `uv sync --extra cua`),
`random` and `rule` (the tool front's baselines).
`build_model(model_name, seed=, rule=)` builds one from the environment; `llm` names the chat model, which `build_model` does not build.

The registered paths, input modalities and question types are summarized in the [decision-model capability matrix](capabilities.md), separately from run evidence and hardware validation.

## The interface

Input: an `Observation(state, images=())` (a JSON object or plain text) and typed questions by name:
`ChoiceQuestion(options, goal=, rules=, operation=)` picks one key among `options` (a string per key, or a dict for
the browser's element rows); `NoulQuestion(question, criteria=)` asks whether a statement holds. Output: a
`Decision(answers, latency_ms, usage, model, raw)` with one `Choice(key, probabilities, confidence)` or
`Noul(p, confidence)` per question name.

`Usage.known` distinguishes reported zero tokens from missing or malformed provider usage. A local backend can
return `Usage()` for a known zero; `Usage.from_payload` preserves tolerant counts but marks incomplete usage unknown.
Input-only decision usage is accepted with zero output tokens. Consumers must not price a partial total as complete.

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
| `jev` | `JevModel(transport)` | `jev` | the request body every front sent before the layer existed, byte for byte; `from_env` picks TypeSafe or the OpenRouter proxy (see [configuration.md](configuration.md)) |
| `clm` | `ClmModel(client, model=)` | `clm` | CLM behind `/v1/systemone`: a frozen Qwen3-8B encoder and two projection heads, scored by an engine that owns everything after the encoder. One POST per decision; `MODEL_SERVICE_CONFIG_ERROR` without `CLM_URL`, `MODEL_CALL_FAILED` when nothing is listening, on an error status, or on a body with no `answers`. Records `url`, the `X-Request-Id` it sent and `served_by` (the served names, read once — by `warm()` on the agent fronts, or on the first decision for `decide`/`probe`); choice and noul ([docs/clm.md](clm.md)) |
| `laya` | `LayaModel(agent, model=)` | `laya` | one forward pass per call on a thread; `MODEL_SERVICE_CONFIG_ERROR` when `input_tokens` fills the window (Laya cuts the state silently; see `LAYA_MAX_LEN` and `LAYA_HEAD_MAX_LEN` in [configuration.md](configuration.md)); `ValueError` and `RuntimeError` from the library become `MODEL_CALL_FAILED` |
| `laya-served` | `ServedLayaModel(client, model=, max_len=)` | `laya-served` | one `POST /v1/systemone` per request to `LAYA_SERVED_URL`, body built with `laya_question()` as for `laya`; one retry on a dropped connection, 502 or 504, and after `Retry-After` on 503, all within `LAYA_SERVED_TIMEOUT_S`; the same window check as `laya`; `model` is the served checkpoint and revision, and `raw` keeps `served_by`, `url`, `request_id` and `server_timing` (see [served-laya.md](served-laya.md)) |
| `cua` | `CuaS1Model(scorer, collator, model=, context_bytes=, option_bytes=)` | `cua` | Cua-S1 Nano, one `score_elements` pass per request on a thread; choice questions only, text only, deterministic; the context is header, state and rules; the checkpoint reads its first 256 bytes, and the first overflowing request logs one warning; `from_env` reads `CUA_S1_*` (see [configuration.md](configuration.md)) |
| `omnijev` | `OmniJevModel(agent, model=)` | `omnijev` | OmniJev (a Qwen3.5 vision-language decision model), one `MSO1.system_one` call per request on a thread; choice and noul questions over the observation's screenshot, deterministic; the text state, goal and rules go in front of each question (a browser state puts its recent actions first and caps the page text at 2,000 characters, so a long page cannot push the history out of the 6,000-character context), a browser element row becomes its label and value; a choice's probabilities, which OmniJev leaves `abstain` out of, are renormalised over the offered keys; an observation without an image is `MODEL_CALL_FAILED`; `from_env` reads `OMNIJEV_*` (see [configuration.md](configuration.md)); in process only: system1-omni has no OmniJev vision worker yet (its supported models are Laya, Open-Jev text and Cua-S1 screenshots), so there is no served `omnijev` path |
| `random` | `RandomModel(seed)` | `random` | uniform over the offered keys, confidence 0, one seeded stream per episode; choice questions only |
| `rule` | `RuleModel(name, rule)` | the rule's name | one-hot, confidence 1; a key outside the menu raises `RuntimeError`, a bug in the rule |

`name` lands in every tick's `source` and in `Episode.policy`; the eval table's columns take their labels from it.
A tool-front tick also keeps `Decision.model` as `model` and, for a served model, `served_by`, `url`, `request_id`
and `server_timing` (`Decision.provenance`).

TypeSafe Jev answers `--model jev`. One request holds a `state` and one or more questions over options the caller
enumerates; the answer holds one option per question, a probability per option and a confidence, from one forward
pass, with no free text. Three heads: `choice` picks one key among the options, `noul` gives the probability that a
statement holds, `score` places the state on an ordered rubric. Input is capped at 32K tokens; endpoint configuration
and API keys are referenced in [configuration.md](configuration.md). Latency and
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
`FakeLayaAgent` in `tests/test_decision_models_laya.py` stands in for the library. The served model runs over an
`httpx.MockTransport` scripted in `tests/test_decision_models_served.py`, answering like the responses recorded from
real servers in `tests/data/served_laya/`. Nothing patches `httpx`.

## Adding a backend

1. `s1a/decision_models/<backend>.py` with `class <Backend>Model(DecisionModel)`: set `name`,
   `supports_images`, `question_types`, `deterministic`; implement `model` and `_decide`, which translates the
   questions and returns a `Reply` whose `answers` are the backend's own dicts; `decide_many` validates them into
   a `Decision`. Keep any heavy import inside `from_env()`.
2. A `case` in `factory.build_model` and the name in `DECISION_MODEL_NAMES`, `tool/loop.py::MODEL_NAMES`,
   `browser/browse.py::BROWSER_MODEL_NAMES`, `rails.RAIL_MODEL_NAMES`, `cli.DECIDE_MODEL_NAMES` and the
   `mcp_server.decide` `Literal`, plus the `match` in `tool/loop.py` and `browser/browse.py` when the backend is
   a decision model; `tests/test_served_laya_registration.py` finds a list that has `laya` without `laya-served`.
3. `tests/test_decision_models_<backend>.py` with `Test<Backend>Contract(DecisionModelContract, IsolatedAsyncioTestCase)`
   plus the backend's mapping tests; a fake for its SDK lives in that file.
4. An optional extra in `pyproject.toml` and an env block in `.env.example` when it needs a dependency.
5. For an in-process backend, time a cold `from_env()` in a fresh process and report the load time in the PR,
   separately from warm inference and episode timing. Profile loads above 5 s. If the loader builds a model
   from its config (`AutoModel.from_config` or a bare `nn.Module`) and then replaces every weight with a
   checkpoint, it can skip random initialization with transformers' `no_init_weights`, but only if loading
   fails on missing keys (`strict=True` or an equivalent check). Prefer fixing this in the model's package;
   Laya added the skip in 0.3.9. This step does not apply to backends served over HTTP.

### Laya browser input shaping

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

The questions need the same care. Laya builds each question's row as `[CLS] instruction [SEP] options [SEP] state`
and fits the instruction and every option into one `head_max_len` budget (192 by default): past it, every option
is cut to an equal share and the instruction to what is left. The browser front's target options are JSON
objects, so a 23-element click head left each option about six tokens, `12: {"element": "[`, and Laya never
saw an element's name. With a folded state, `laya_browser_question` rewrites each browser question: the
instruction becomes the goal and the operation (the agent's rules, 446 tokens and about 550 on a target head, are
dropped), and each target option becomes its element's label and value, `Where from? = Zurich`. Each head keeps its own ask (`text_value` asks which value to type, not which operation comes next); two options that shorten to the same text keep their key in front (`[2] Select flight: Swiss LX 318,`); a blocked row keeps its overlay's name (`(blocked by Cookie consent)`). Measured on a
Google Flights run with this shape: 157 to 206 tokens for the operation head, 73 to 101 for the TYPE_TEXT and
PRESS_ENTER heads, 92 to 915 for the CLICK head (a calendar page offers 66 days), and 98 to 1,002 tokens of folded
state. Browser runs therefore want `LAYA_MAX_LEN=1536` and `LAYA_HEAD_MAX_LEN=1024`.

Both folds make the request fit; they do not make the stock checkpoint drive a web form. On 2026-09-28, with both
on and those two settings, `s1a run flights --model laya` on CPU answered DONE at the first step (DONE 0.55, CLICK
0.23; 12.4 s, 1,254 input tokens over four questions) where Jev needs twelve steps. Replayed offline over Jev's
twelve recorded steps of the same task, the checkpoint picks Jev's answer on 4 of 23 questions (the operation head
and the chosen operation's target head). Laya's model card names email triage, routing, guardrails and moderation
as what its checkpoints are for; browser use would need a checkpoint fine-tuned on browser steps.
