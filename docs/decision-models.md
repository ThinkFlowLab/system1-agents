# The decision-model layer: one interface between every front and the decision model

Code: `s1a/decision_models/`. Tests: `tests/test_decision_models_*.py`, the shared contract in
`tests/decision_model_contract.py`. The wire itself (the async HTTP client, the `typesafe` and `openrouter` backends)
stays in `s1a/decision_models/wire.py` and reads no answer.

Every front that asks "which one" (the tool loop, the browser policy, the rails, `decide`, `probe`, the MCP server)
talks only to `DecisionModel`; the wire client is private to `s1a/decision_models/`. A decision model is a classifier over options the caller enumerates. It
reads a state and returns a distribution over the offered keys. Hugging Face writes "System 1 decision model" and
TypeSafe "System One model". This repository uses the terms interchangeably. `--model` picks the backend: `jev` (TypeSafe Jev over HTTP),
`jev-vl-served` (JEV-27B-VL over System1-Omni's native flat HTTP protocol, browser path only),
`clm` (CLM's `clm-serve` over HTTP, see [clm.md](clm.md)),
`laya` (in process, behind `uv sync --extra laya`), `laya-served` (Laya served over HTTP by system1-omni, see
[served-laya.md](served-laya.md)), `cua` (Cua-S1 Nano in process, behind `uv sync --extra cua`),
`onejev` (OneJev over HTTP via `qev`/`flide` with a screenshot), `random` and `rule` (the tool front's baselines).
`build_model(model_name, seed=, rule=)` builds one from the environment; `llm` names the chat model, which `build_model` does not build.

The registered paths, input modalities and question types are summarized in the [decision-model capability matrix](capabilities.md), separately from run evidence and hardware validation.

## The interface

Input: an `Observation(state, images=())` (a JSON object or plain text) and typed questions by name:
`ChoiceQuestion(options, goal=, rules=, operation=)` picks one key among `options` (a string per key, or a dict for
the browser's element rows); `NoulQuestion(question, criteria=)` asks whether a statement holds. Output: a
`Decision(answers, latency_ms, usage, model, raw)` with one `Choice(key, probabilities, confidence)` or
`Noul(p, confidence)` per question name. **`confidence` always means the same thing on every backend:** for a
choice it is `max_k p_k` (the peak probability); for a noul it is `max(p, 1 - p)`. Validation computes both from
the validated probabilities in `s1a/decision_models/validation.py`, so Laya's normalised entropy, Jev's normalised
margin, and OmniJev's recompute never become the reported `confidence`. Each backend's own number is kept as
`backend_confidence` on the answer and under `Decision.raw["backend_confidence"]` (a map of question name → number)
when the backend sent one in [0, 1]. These numbers are **not** calibrated unless a backend says so; do not reuse a
threshold across backends.

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
within 0.02 and peak at the key; a noul probability lies in [0, 1]. After those checks, `Choice.confidence` is set
to `max_k p_k` and `Noul.confidence` to `max(p, 1 - p)` regardless of what the backend put in its `confidence` field.
An unusable answer is re-asked with the same request up to `attempts` times (never on a deterministic backend, never
after a transport error) and then raises `MODEL_CALL_FAILED`. `decide`, `choose` and `ask` are one-question
shorthands; `warm()` and `close()` open and release the backend.

## Backends

| `--model` | class | `name` | notes |
|---|---|---|---|
| `jev` | `JevModel(transport)` | `jev` | the request body every front sent before the layer existed, byte for byte; `from_env` picks TypeSafe or the OpenRouter proxy (see [configuration.md](configuration.md)) |
| `jev-vl-served` | `ServedJevVLModel(url=, timeout_s=)` | `jev-vl-served` | `autotrust/JEV-27B-VL` through native flat `POST /v1/systemone`; exactly one current PNG/JPEG, choice and noul only; sequential questions share one deadline, with no HTTP retry or answer re-ask. Browser and direct factory paths; local tests use mocks, with separate [bounded R3 GPU/browser validation](#served-jev-vl). |
| `clm` | `ClmModel(client, model=)` | `clm` | CLM behind `/v1/systemone`: a frozen Qwen3-8B encoder and two projection heads, scored by an engine that owns everything after the encoder. One POST per decision; `MODEL_SERVICE_CONFIG_ERROR` without `CLM_URL`, `MODEL_CALL_FAILED` when nothing is listening, on an error status, or on a body with no `answers`. Records `url`, the `X-Request-Id` it sent and `served_by` (the served names, read once — by `warm()` on the agent fronts, or on the first decision for `decide`/`probe`); choice and noul ([docs/clm.md](clm.md)) |
| `laya` | `LayaModel(agent, model=)` | `laya` | one forward pass per call on a thread; `MODEL_SERVICE_CONFIG_ERROR` when `input_tokens` fills the window (Laya cuts the state silently; see `LAYA_MAX_LEN` and `LAYA_HEAD_MAX_LEN` in [configuration.md](configuration.md)); `ValueError` and `RuntimeError` from the library become `MODEL_CALL_FAILED` |
| `laya-served` | `ServedLayaModel(client, model=, max_len=)` | `laya-served` | one `POST /v1/systemone` per request to `LAYA_SERVED_URL`, body built with `laya_question()` as for `laya`; one retry on a dropped connection, 502 or 504, and after `Retry-After` on 503, all within `LAYA_SERVED_TIMEOUT_S`; the same window check as `laya`; `model` is the served checkpoint and revision, and `raw` keeps `served_by`, `url`, `request_id` and `server_timing` (see [served-laya.md](served-laya.md)) |
| `cua` | `CuaS1Model(scorer, collator, model=, context_bytes=, option_bytes=)` | `cua` | Cua-S1 Nano, one `score_elements` pass per request on a thread; choice questions only, text only, deterministic; the context is header, state and rules; the checkpoint reads its first 256 bytes, and the first overflowing request logs one warning; `from_env` reads `CUA_S1_*` (see [configuration.md](configuration.md)) |
| `cua` (served 4B) | `ServedCuaFourBModel(client)` | `cua` | `CUA_S1_VARIANT=4b` with `CUA_S1_URL`: Cua-S1 4B 0.2 multimodal on system1-omni's worker, one `POST /v1/systemone` per request with the observation's one screenshot as a PNG/JPEG data URL and up to 8 choice questions of at most 26 options; a screenshot above 1,048,576 pixels or 2,048 per side is scaled down (pixel targets are fractions, so the point does not move); each question's instructions are the goal, state and rules cut to the worker's 16,384 characters, without Qwen's vision control tokens; an answer must cover exactly the offered keys, sum to one and choose its peak, else `MODEL_CALL_FAILED`; the same retry and error mapping as `laya-served`; `raw` keeps `served_by`, `url`, `request_id`, `server_timing` and the screenshot's SHA-256 and size, never its bytes |
| `omnijev` | `OmniJevModel(agent, model=)` | `omnijev` | OmniJev (a Qwen3.5 vision-language decision model), one `MSO1.system_one` call per request on a thread; choice and noul questions over the observation's screenshot, deterministic; the text state, goal and rules go in front of each question (a browser state puts its recent actions first and caps the page text at 2,000 characters, so a long page cannot push the history out of the 6,000-character context), a browser element row becomes its label and value; a choice's probabilities, which OmniJev leaves `abstain` out of, are renormalised over the offered keys; an observation without an image is `MODEL_CALL_FAILED`; `from_env` reads `OMNIJEV_*` (see [configuration.md](configuration.md)); in process only: system1-omni has no OmniJev vision worker yet (its supported models are Laya, Open-Jev text and Cua-S1 screenshots), so there is no served `omnijev` path |
| `onejev` | `OneJevModel(client, model=)` | `onejev` | OneJev (OmniJev org, Apache-2.0) served by `qev`/`flide` over TypeSafe-compatible `POST /v1/systemone` plus a `media` field; choice and noul over the observation's screenshot(s), deterministic; state carries `<image:N>` markers and each screenshot is a PNG/JPEG/WebP data URL in `media` (digest and size in `raw.sent_images`, never the bytes); an observation without an image is `MODEL_CALL_FAILED` (same policy as `omnijev`); one refused-connection / 502 / 504 retry within `ONEJEV_TIMEOUT_S`; `raw` keeps `served_by`, `url`, `request_id` and `server_timing`; `from_env` reads `ONEJEV_*` (see [configuration.md](configuration.md)); browser and `build_model` only — not tool/rails/`decide`. Local tests use a stub server; live GPU validation is **not recorded** in this checkout |
| `random` | `RandomModel(seed)` | `random` | uniform over the offered keys (contract confidence = `1/|options|`), one seeded stream per episode; choice questions only |
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


### Served OneJev

`onejev` is registered in the browser front and `build_model()`. Point `ONEJEV_URL` at a local
[`qev`](https://github.com/OmniJev/OneJev) or [`flide`](https://github.com/OmniJev/Flide) server that speaks
`/v1/systemone` with `media`. Example smoke (not run in CI; needs a GPU or GGUF build):

```bash
# terminal 1 — install from the OneJev repo (pip install qev is a different package)
qev serve --model OmniJev/OneJev1.6-4B --port 8000
# or: flide serve --model OmniJev/OneJev1.6-4B --port 8000

# terminal 2
ONEJEV_URL=http://127.0.0.1:8000 ONEJEV_TIMEOUT_S=60 \
  uv run s1a run flights --model onejev --rethink off
```

This is a setup recipe only. Capability registration and stub-server contract tests do **not** establish task
success rates or hardware validation; see [capabilities.md](capabilities.md).

### Served JEV-VL

`jev-vl-served` is registered in the browser front and `build_model()`. It sends the observation's text state,
goal, operation and rules with exactly one current PNG/JPEG to a native System1-Omni frontend serving
`autotrust/JEV-27B-VL`. The original image bytes are sent without resizing or truncation. Limits are 4 MiB per
image, 2,048 pixels per side, 1,048,576 pixels in total, and 8 MiB for each complete UTF-8 JSON body.
A missing, invalid or oversized screenshot fails the decision instead of falling back to text.

Choice and noul questions are sent sequentially under one `S1A_DECISION_TIMEOUT_S` deadline (5 s by default).
HTTP errors and invalid flat responses fail the whole decision. No partial action is executed, and the adapter
does not retry HTTP requests or re-ask invalid answers. A choice with one offered key is forced locally after
image validation, with zero model requests and zero usage; it is not a learned choice. `latency_ms` covers the
whole client decision, including preparation and validation, rather than GPU inference alone.

Browser ticks retain `decision_raw` only for `jev-vl-served`; the existing served backends keep their
provenance fields without adding a full response payload. JEV-VL response records keep only `server-timing`,
`x-request-id`, `content-type` and `content-length` headers. Forced choices have a null per-response
`request_id`, and an entirely forced decision also has a null top-level `request_id`, because no HTTP request
was sent.

For a ready frontend, a browser viewport within the image limits, and the usual browser chat-model configuration,
the invocation is:

```bash
JEV_VL_URL=http://127.0.0.1:8080 S1A_DECISION_TIMEOUT_S=30 uv run s1a run flights --model jev-vl-served --rethink off
```

This is a setup example; the validation below does not establish Flights task completion. Other agent fronts
are not registered for this image-required backend. This path is separate from the in-process `omnijev` adapter.

Bounded R3 validation (2026-10-14) used frozen Agents `0def92771746ec94adc38bf41be9720bccfbf19d`
and Omni `26dc76bf65c544815e7b285866791da1e703816d` on one NVIDIA H800. Native correctness
(612 requests), cache eviction (36 requests), and the full 60-episode paired HTTP replay (1,080 timed requests)
passed their checks. The [RECOVERABLE fixture](../evals/recovery/fixture.py), run through the
[live browser evaluator](../evals/recovery/live.py) with recovery off, completed three actions and a verified
`hello world` POST. Four operation choices were non-forced; six forced targets do not count as learned choices.
A separate chat/value model supplied text; JEV-VL selected the operations. The same-trial application video
and Agents/Omni evidence replay contain 294 frames over 11.76 seconds. R1 failed in transport; R2 completed
the task but had an empty recording; R3 passed the complete evidence checks. All three attempts are retained.
This is a single-fixture smoke check, not broad task-quality evidence or an end-to-end browser speedup claim.

Run-local audit records remain outside this repository:
`20261012T114800Z-online-vision-agents/revisions/26dc76b-agents-0def927`, browser job `419202`.
The same-trial [evidence replay](https://github.com/linear3735/system1-agents/releases/download/jevvl-r3-419202-evidence/evidence-replay.mp4)
and [sanitized R3 records](https://github.com/linear3735/system1-agents/releases/download/jevvl-r3-419202-evidence/jevvl-r3-evidence.zip)
are public; the full GPU raw records are not published. The evidence replay SHA-256 is
`58550ee293bbfb9c5b1f563c03e8cbdb6f6ac2a3ef23080f73ec3435b932a9e4`.

## What each front sends

- Tool front (`ToolDecisionModel`): one `pick` choice question over the candidates with the agent's rules; the state
  includes the plan and the harness notices when the rethink rail wrote them.
- Browser front (`BrowserDecisionModel`): `operation`, one `<op>_target` per head (CLICK, TYPE_TEXT, SELECT) and
  `text_value` when goal values are offered, all in one `decide_many(attempts=2)` over the page state; `interpret`
  reads the validated decision onto a candidate. Ticks keep `decision_ms` and `input_tokens` for the artifact readers.
- Rails (`DecisionModelRail`): one `check` question, noul or choice, banded by the spec's thresholds.
- `decide`, `probe` and the MCP server: one `pick` question; the printed dict stays `choice`, `probabilities`,
  `confidence` (= `max_k p_k`), `ms`, and `backend_confidence` when the backend sent its own number.

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
2. A `case` in `factory.build_model` and the name in `DECISION_MODEL_NAMES`, then register only fronts that
   provide the backend's required input. Update their model-name list and dispatch `match`, with registration
   tests. Text backends may also need `tool/loop.py::MODEL_NAMES`, `rails.RAIL_MODEL_NAMES`,
   `cli.DECIDE_MODEL_NAMES` and the `mcp_server.decide` `Literal`; image-required backends need an actual image
   path. `tests/test_served_laya_registration.py` checks Laya's registrations across its supported fronts.
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
