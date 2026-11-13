# Browser steps: offline step-level scoring (#109)

`benchmarks.legs.browser_steps` asks a decision backend one browser step at a time from
[Mlchemist/myjev-browser-decisions](https://huggingface.co/datasets/Mlchemist/myjev-browser-decisions)
(revision `8c0c5d532458f7ba25c9684b267694848e815d5a`) and scores the answers against the gold operation and
element keys. It runs offline in minutes and needs no live site.

**Step success is not task success.** A step is scored with the gold page state and gold history, so a backend that
gets most steps right can still fail whole tasks, and this does not replace the repeatable task suites of #13.

## Licence and data handling

The dataset is **CC BY-NC-SA 4.0** (it includes WebLINX), and its Mind2Web rows come from authors who ask that the
data not be republished. No dataset row is committed here: `fetch` downloads the pinned files at run time and
keeps a file only when its hash matches (SHA-256 for the two splits, the git blob id for the id list). Results
directories hold per-step answers keyed by the dataset's step id; keep them outside the repository and do not
publish the saved requests, which contain page text.

## Run

```bash
python -m benchmarks.legs.browser_steps fetch --out /data/myjev \
  --split benchmark --split calibration --split jev_answered_ids
python -m benchmarks.legs.browser_steps predict --data /data/myjev/benchmark.jsonl --split benchmark \
  --model jev --run-meta /path/environment.json --out /results/jev-benchmark
python -m benchmarks.legs.browser_steps score --run /results/jev-benchmark \
  --data /data/myjev/benchmark.jsonl --ids /data/myjev/benchmark_jev_answered_ids.txt
python -m benchmarks.legs.browser_steps predict --data /data/myjev/calibration.jsonl --split calibration \
  --model jev --out /results/jev-calibration
python -m benchmarks.legs.browser_steps thresholds --run /results/jev-calibration \
  --data /data/myjev/calibration.jsonl
```

`--model` takes any text backend `build_model` knows (`jev`, `laya`, `laya-served`, `cua`, `clm`, `random`), configured
from the environment as for the agents, so a local System One server is `--model jev` with `TYPESAFE_API_URL`.
Screenshot backends are out of scope: the set has no images. `--run-meta` is copied into `run.meta.json` next to
the dataset revision and hash, the request shape, the backend and model id and the Agents SHA; put the backend
revision, hardware and `S1A_DECISION_TIMEOUT_S` there.

The collector sends one request per step, in dataset order, through `DecisionModel.decide_many` with one attempt
and no retries. A transport error, a timeout or an answer that fails validation is saved with its error and is
**wrong** on every metric; steps not attempted (`--limit`) stay in the denominator of the full-split numbers
(`score --limit N` adds the attempted prefix separately).

## Metrics

`score` prints, for the full split and for the Jev-answered subset (`--ids`), each with numerator and
denominator:

- operation accuracy, over all steps;
- element accuracy, over steps whose gold operation takes a target: the answer to the gold operation's target
  question is one of the gold keys, whatever operation was chosen (Mind2Web's element accuracy);
- step success: the operation is right and, when it takes a target, so is the element;
- invalid or failed answers;
- Brier score and 10-bin ECE of the operation confidence (`max p`) against operation correctness, and the
  multiclass Brier of the operation probabilities;
- p50/p95 of the backend's decision latency;
- step success by gold operation.

`thresholds` reads a calibration run and prints, per operation-confidence threshold from 0 to 0.95, the share of
steps the backend keeps, their step success and the share deferred (failed steps are always deferred). This is
the per-backend table the deferral of #94 / #107 needs; the calibration split shares no website with the
benchmark split.

## Request shape: theirs and ours

The dataset's request is what jev-ultrafast sends. `--shape dataset` re-sends it unchanged: over the 652
calibration records, the body `JevModel` builds from the converted questions is identical to the dataset's
(checked on the real file; the tests pin the same property on a synthetic fixture). `--shape ours` (the default)
keeps the state, the question names and the element rows and swaps in the text `BrowserDecisionModel` sends.

Checked on the calibration split, the structure already matches our action space:

| | dataset | our agent |
|---|---|---|
| top level | `model`, `state`, `questions` | same |
| `state` | `page` {url, title, text}, `elements`, `recent_actions` (last 10, `page_changed` blanked) | same keys |
| element rows | `index`, `role`, `label`, `value`, `operations`, optional `checked`/`expanded`/`selected`/`options` | same, plus `region`, `blocked_by`, `click_did_nothing` when the probe sets them |
| questions | `operation`, `click_target`, `type_text_target`, `select_target` | same names; plus `press_enter_target` and `text_value` when offered |
| target options | `element`, `current_value`, `role`, optional state keys | same keys |
| SELECT keys | `index:option` | same |

What `--shape ours` changes: the operation descriptions (our `_OPERATION_LABELS`) and the rules (our
`OPERATION_RULES["en"]`, and `TARGET_RULES["en"]` after it on target questions). What it cannot reproduce: our
`PRESS_ENTER` head and `text_value` question (the dataset has no such gold), our `option` field on SELECT rows
(the dataset folds the option label into `element` as `label → option`), and our probe's `region` /
`blocked_by` rows. Scores therefore measure the model on our wording over jev-ultrafast's page snapshots, not on
our probe.

## Results

| Backend | Split | Shape | Steps | Operation acc. | Element acc. | Step success | Failed | Op. conf. Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| `random` (seed 0), floor | calibration | ours | 652 | 98/652 (15.0%) | 51/608 (8.4%) | 12/652 (1.8%) | 0/652 | 0.127 | 0.008 |
| `random` (seed 0), floor | calibration | dataset | 652 | 98/652 (15.0%) | 51/608 (8.4%) | 12/652 (1.8%) | 0/652 | 0.127 | 0.008 |

The random rows were run offline on 2026-11-02 at Agents SHA `69527d9` (base of this change) with the code in
this directory; they only show the collector and scorer working end to end on real records and give the chance
floor. Hosted Jev and an open backend on the benchmark split have **not been run yet**: they need a TypeSafe key
and a GPU host, and their rows will be added here with backend revision, hardware and Agents SHA.
