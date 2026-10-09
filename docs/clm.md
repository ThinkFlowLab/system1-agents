# CLM as a decision model: `--model clm`

CLM is a second kind of System 1 model: **the engine does not compute embeddings.** A frozen Qwen3-8B encoder
runs as its own process behind an OpenAI-compatible `/v1/embeddings` endpoint, and everything after it — two
projection heads, the cosine score, the temperature and the typed answer — is the engine. That split is why the
client here is an HTTP client, and why `clm-serve` needs an embeddings server beside it before it can answer
anything.

CLM's `POST /v1/systemone` follows the same wire schema as the other served models, so this backend is a sibling of
[`laya-served`](served-laya.md) rather than a new kind of thing: the difference is the engine behind the URL, not
the interface. Upstream ships the server (`clm-serve`), so system1-agents needs no CLM extra and no torch.

## Setup

Two processes, and the second one needs the first. `system1-omni`'s
[`recipe/clm/README.md`](https://github.com/ThinkFlowLab/system1-omni/blob/main/recipe/clm/README.md) is the full
version, including a CPU stub that stands in for the encoder when only the plumbing is being checked:

```sh
# 1. the encoder. `serve_qwen3_8b.sh` from the CLM checkout is vLLM with last-token pooling, which is what the
#    heads were trained against; any server matching it will do.
GPU=0 PORT=8090 UTIL=0.35 ./serve_qwen3_8b.sh

# 2. the engine. CLM_CKPT is the file; without it clm-serve looks in ~/.cache/clm and downloads.
CLM_CKPT=/path/CLM_v0.1-8B.pt clm-serve --port 8091 \
  --emb-url http://127.0.0.1:8090/v1/embeddings --emb-model qwen3-8b
```

```sh
CLM_URL=http://127.0.0.1:8091 uv run s1a run ticket_router --model clm --rethink off --episodes 1 --seed 0
```

Point `CLM_URL` at system1-omni's frontend (`omni-jev`, `:8080`) or straight at `clm-serve` (`:8091`); the
frontend routes `/v1/systemone` and `/health`, which is all this backend reads, so both work. Going through the
frontend is the path this repository's guidance asks for and the one the recorded demo uses.

`CLM_URL` is the only setting that is required. `CLM_MODEL` picks `clm-latest` (the default) or `clm-raw`, the
ablation that scores in the raw encoder space with no projection head — useful as a control, since it needs no
training to be meaningful. The rest are in [configuration.md](configuration.md).

## What it sends

`clm_question` builds the body `clm.client` builds, which is what `clm-serve`'s own docstring documents:

```json
{"type": "choice", "instructions": "route it", "criteria": {"billing": "Charges and refunds"}}
{"type": "noul", "instructions": "Does the customer ask for a refund?"}
```

Two things differ from the Jev-shaped body, and both follow from how CLM reads a request rather than from taste:

- **`instructions` is text.** The state head embeds `state` and `instructions` together, separated by a blank line,
  and the heads were trained on a short state and a short question (`"Choose the best safe action for the
  dinosaur."`). Jev's `{goal, operation, rules}` object is not what this model reads, so the parts are joined into
  one string.
- **A choice's `criteria` is `{key: description}`**, and the action head embeds each description as the candidate's
  own text with nothing prefixed. A description need not be a string: `to_text` renders an object as `field: value`
  lines — top-level fields separated by a blank line, nested ones indented — and `null` as nothing. The browser
  front's `<op>_target` heads describe each element as `{element, current_value, role, checked, ...}`
  (`s1a/browser/action_space.py`), so those fields are what the head embeds;
  `tests/data/clm/browser_target.json` records one such request and the answer `clm-serve` gave it.

## What it supports, and what it does not

Choice and noul, like the other served backends; a `score` question is not offered to it. Images are not read. The
engine's decision is stable to six decimals for the same request, so the model is `deterministic` and an unusable
answer is not re-asked — a second call would pay the encoder again for the same distribution. Every decision records
the URL, the `X-Request-Id` it sent and what `/health` reported — the names served, whether the engine has its
encoder, the device its vector cache sits on, and `read_at`, when that one reading was taken, because a frontend
restarted mid-run is not noticed; the agent fronts call `warm()` for that read, and `decide` and `probe` make it once
on their first decision instead. `server_timing` carries the response's `Server-Timing` plus the engine's own
`X-CLM-Latency-Ms` under `clm`, which is the number that separates the engine's scoring time from the round trip
`Reply.latency_ms` measures.

The engine keeps a candidate-vector cache across requests, which is the feature that makes it interesting and also
the reason `usage.input_tokens` counts only encoder cache misses: the field moves with cache state rather than with
what the request asked for. system1-omni's `recipe/compare_with_backend.py` compares the `answers` subtree for that
reason.

## What it measured

On [ticket routing](../evals/ticket_router/CLM.md): **18/90 = 20%** with the tool loop's framing, against 51/90
for the keyword baseline, 63/90 for Laya and 17/90 for uniform random — and every one of the three seeds gave
exactly 6, because CLM answers `human` for all thirty tickets. `s1a probe evals/ticket_router/probe.jsonl --model
clm` returns `4/12 (33%)`, **`not a decision-model task`**, against the repository's own 80% threshold, where the
repository's keyword baseline scores 8/12 on the same cases.

The evidence page has the mechanism, which is measured rather than argued: the loop puts a 696-character rules
essay into `instructions`, identical for every ticket, and it raises the least alike pair of ticket embeddings from
+0.81 to +0.96 — thirty tickets the heads are then asked to tell apart. Feeding CLM a short question instead of the
essay reaches 36–39/90, still below the keyword baseline. CLM is wired up and works; it is not the model to pick
for this task yet.
