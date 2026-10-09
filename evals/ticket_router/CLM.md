# Ticket router: CLM as the decision model

Date: 2026-10-06. The plan below was fixed before the runs. This page reports one measurement and one mechanism;
it is not a benchmark, and `--episodes 1 --seed 0` is 30 decisions, not 90.

## Setup

| | how | what answered |
|---|---|---|
| CLM | `--model clm`, `CLM_URL=http://127.0.0.1:8091` | `clm-latest` on `CLM_v0.1-8B.pt`, `clm-serve` over a tunnel |
| encoder | `serve_qwen3_8b.sh`'s settings, in Transformers | Qwen3-8B, last-token pooling, bfloat16, on one RTX 4090 (compute capability 8.9, driver 595.71.05, CUDA 13.0, transformers 5.17.0) |

- `system1-agents` at the commit that adds `--model clm`; `system1-omni` at `0450083`, whose
  `recipe/clm/native/VALIDATION.md` records the engine's own agreement with CLM's reference.
- `s1a run ticket_router --model clm --rethink off --episodes 3 --seed 0`, the whole 30-ticket set three times.
- The checkpoint is the published one, sha256 `b2b4a8c9…`; the engine reported `clm-latest` and the URL in every
  tick's `served_by`.
- 90 decisions, median 168 ms, no invalid keys, no errors.

## Results

| strategy | correct | source |
|---|---:|---|
| uniform random | 17/90 (19%) | [RESULTS.md](RESULTS.md) |
| keyword rule baseline | 51/90 (57%) | [RESULTS.md](RESULTS.md) |
| in-process Laya | 63/90 (70%) | [SERVED_LAYA.md](SERVED_LAYA.md) |
| **CLM, this loop's framing** | **18/90 (20%)** | this page |
| CLM, a short question instead of the rules | 36/90 (40%) | this page |

`--episodes 3 --seed 0` is 90 decisions over the same 30 tickets, and **every seed gave exactly 6**: the number is
not a noisy estimate, it is what a collapsed predictor scores. CLM's six are the six tickets whose label is
`human` — it answered `human` for all thirty, at a confidence of 0.86–0.98.

## The repository's own fit probe

`s1a probe evals/ticket_router/probe.jsonl` is the instrument this repository ships for exactly this question: twelve
hand-written cases, and a verdict at a stated threshold (`FIT_THRESHOLD = 0.8` in `s1a/probe.py`). `RESULTS.md`
records that it was never run — "the English fit probe was attempted but failed with `decisions connection failed`"
— and that English model quality therefore remains unvalidated.

With CLM it runs, and this is its verdict:

```
CLM_URL=http://127.0.0.1:8091 uv run s1a probe evals/ticket_router/probe.jsonl --model clm
right: 4/12 (33%)   median ms: 197   verdict: not a decision-model task
```

It answers `human` for all twelve and its four correct are the four whose `accept` includes `human` — the same
collapse as the batch, on a set written to probe the hard cases (negation, resolved background, an explicit human
request).

The threshold is not the probe being impossible: the repository's own keyword baseline scores **8/12 (67%)** on the
same twelve cases, twice CLM's four, with the same collapse on the cases that need the negation read
(`negation`, `background keyword`, `negated background`, `background payment` all go to `human`). Neither reaches
80%, so the verdict is a gate rather than a ranking — but it is the gate this repository chose, and CLM does not
pass it.

## Why, and what it costs

The tool loop observes `{"ticket": {…}, "progress": {…}}` and asks `ChoiceQuestion(queues, rules=RULES)`; the
backend sends the observation as the state and `RULES` as the instructions, which is what the wire body on this
branch shows. CLM's state head embeds the two together, and `RULES` is 696 characters of routing rubric that is
identical for every ticket — so it dominates that text and the thirty tickets stop being distinguishable. `RULES`
also ends with "If no unique queue fits, choose human", which is the answer the collapse lands on.

That the rules crowd the tickets out is measured, not assumed. Embedding the thirty tickets' state texts and taking
every pairwise cosine (435 pairs, one RTX 4090):

| the text the encoder sees | mean | lowest pair | highest pair |
|---|---:|---:|---:|
| the ticket alone | +0.9327 | +0.8135 | +0.9938 |
| the ticket, then the rules | +0.9893 | +0.9636 | +0.9974 |

The rules raise the *floor* from +0.814 to +0.964: the two least alike tickets in the set end up more similar than
almost any pair was before. Whatever the heads do on top of that, they are being asked to separate thirty texts
that have become nearly the same text.

Sending the same tickets, the same five queue descriptions and the same rules under different renderings gives
(`evals/ticket_router/compare_framings.py`, 3 seeds, 90 decisions per row):

| state | instructions | correct |
|---|---|---:|
| the observation as JSON | the rules | 18/90 |
| the ticket as one sentence, then the rules | nothing | 18/90 |
| the ticket as one sentence, then the rules | a short question | 36/90 |
| the ticket as one sentence | a short question | 36/90 |
| the ticket as one sentence | nothing | 39/90 |

All five rows are in `compare_framings.py`; the third one is worth pointing at, because it differs from the second
by nothing but a question this backend may not write, and that is the whole of the difference between 18 and 36.

**These four rows are exploratory.** Twelve correct of thirty against six is Fisher exact p = 0.158 — a trend,
not a result — and every interval here belongs to n=30. At the observed rates, about **sixty** tickets would put
the first-against-third gap at p ≈ 0.03; thirty cannot. The probe above is the part of this page that carries a
verdict, because a threshold is not a significance test.

The second row is the one that matters for where a fix belongs. Moving the rules out of `instructions` and into
the state — which is what the `cua` backend does with `cua_context`, and the only one of these a backend could do
without inventing text — changes **nothing**: 18/90 either way. Every point of the gap comes from replacing the
696-character rules essay with a short question, and that sentence is the caller's to write. A backend that made
it up would be answering a question nobody asked.

CLM's own preference — the short state and the short question its examples use (`"Dino runner game. 2 large cacti
ahead, 96 px away."` / `"Choose the best safe action for the dinosaur."`) — is the 40% row, and it is still below
the keyword baseline. `clm-raw`, the ablation that scores in the raw encoder space with no projection head, picks
`account` for the obvious password ticket where `clm-latest` picks `returns`, on two machines, so the encoder
ranks that ticket correctly and the heads are what move it (n=1; a control, not a finding).

## Recording

`docs/assets/demos/ticket-router-clm.gif` is one run through the path
[CONTRIBUTING.md](https://github.com/ThinkFlowLab/system1-agents/blob/main/CONTRIBUTING.md#agent-video-demos)
asks for, recorded at the terminal by `demo.sh` and played at 1x: the System1-Omni frontend ready and saying who
is behind it, the tickets, the queues the model chose, and the identity recorded in each tick.

```
curl -s http://127.0.0.1:8080/health      # omni-jev, with clm-serve behind it
  {"ok": true, "embedder": true, "models": ["clm-latest", "clm-raw"]}   vector cache on cuda

CLM_URL=http://127.0.0.1:8080 s1a run ticket_router --model clm --rethink off \
  --episodes 1 --seed 0 --showcase --log
  30 decisions, every one `human`, median 123 ms   mean_score 6.0
  served_by {"url": "http://127.0.0.1:8080", "models": ["clm-latest", "clm-raw"],
             "embedder": true, "device": "cuda", "source": "health"}
  answered {'human': 30}    correct 6/30 routed as labelled
```

The clip runs **all thirty**, not a slice: at 123 ms a decision the whole set costs four seconds, which fits the
twenty-to-forty-five seconds the guidance asks for, and a slice of three would show nothing a reader can use.

The topology, which the caption on the pull request states in full: `s1a` -> `omni-jev` (system1-omni's frontend,
`:8080`) -> `clm-serve` (upstream CLM's own server, `:8091`) -> Qwen3-8B on one RTX 4090. **`omni-clm`, the engine
this project tracks, does not serve HTTP** — `src/models/clm/README.md` says the frontend owns that socket — so the
engine behind `omni-jev` here is upstream's, not this repository's. That is the same gap
[`docs/supported-models.md`](https://github.com/ThinkFlowLab/system1-omni/blob/main/docs/supported-models.md)
describes, and it is why the three tickets in this clip all route to `human`.

Nothing in the clip is typed, reordered or invented: the frames are drawn from the output as it arrived. The same
run as an asciicast is `ticket-router-clm.cast`. A demo illustrates one run; the probe's verdict and the tables
above are the evidence, and neither is a benchmark.

## Reproduce

```sh
# the encoder and the engine, per system1-omni's recipe/clm/README.md
CLM_URL=http://127.0.0.1:8091 uv run s1a run ticket_router --model clm --rethink off --episodes 1 --seed 0
```

The run's job folder holds the per-ticket routes, the probabilities behind each one and `served_by`.

## Limits

- **30 independent tickets, and no more.** Three seeds shuffle the same thirty, so 90 decisions is not 90 samples
  and the per-seed counts are identical rather than merely close. Every interval below is therefore computed on
  n=30; reading the seeds as 90 samples would halve it and claim precision that is not there.

  | | rate | 95% interval (n=30) |
  |---|---:|---|
  | CLM, this loop's framing | 20.0% | 9.5–37.3% |
  | uniform random | 20.0% | 9.5–37.3% |
  | CLM, a short question | 40.0% | 24.6–57.7% |
  | keyword rule baseline | 56.7% | 39.2–72.6% |
  | Laya | 70.0% | 52.1–83.3% |

  CLM's interval and random's are the same interval, so the defensible reading is "not distinguishable from
  random", not "exactly random". The short-question row's upper bound reaches the keyword baseline, so that row is
  not separated from it with confidence either.
- **Repeating the run does not narrow any of this**, and that is measured. Through `omni-jev`: the 90-decision
  protocol gave `mean_score 6.0` three times, the single-episode run gave 6.0 three times, and the fit probe
  returned `4/12, not a decision-model task` twice. Only the client round trip moved (78, 82, 85 ms median). The
  engine is deterministic — the same request five times gives byte-identical probabilities — so a repeat returns
  the same number rather than a better estimate of it. More seeds shuffle the same thirty tickets. The only thing
  that narrows these intervals is a larger labelled set: 300 tickets at these rates would give 16–25% and 35–46%.
- **Every number on this page was taken straight against `clm-serve`**, and those repeats are the check that the
  frontend does not change them: `omni-jev` proxies `/v1/systemone` unchanged, so the figures are the same through
  `http://127.0.0.1:8080` as through `http://127.0.0.1:8091`.
- The encoder is Transformers with last-token pooling, not a vLLM pooling server. The heads were trained against
  the vLLM path; a deployment's encoder is a different implementation and this page does not measure it.
- The framing table is one run per row and was measured after the first result, so it is exploratory.
- CLM answers choice and noul; the ticket router needs choice only.
