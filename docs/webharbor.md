# WebHarbor seeded slice

[WebHarbor](https://github.com/aiming-lab/WebHarbor) ([arXiv:2611.03634](https://arxiv.org/abs/2611.03634)) ships
Dockerized reconstructions of real websites with seeded databases, a reset to the seeded state, and per-site
**deterministic verifiers** that check the final answer, the visited pages and the database changes. That gives the
browser front what live WebVoyager cannot: repeatable starting conditions and an independent success check (#13,
#118). It is the offline counterpart of the live Online-Mind2Web slice (#116).

## The slice

`evals/webharbor/slice.json` pins 12 Allrecipes tasks of `webharbor-webvoyager@1.0`
([`ChilleD/WebHarbor-Tasks`](https://huggingface.co/datasets/ChilleD/WebHarbor-Tasks) at revision
`3b1da40d1f52426706aab38ae7cbd1a6dc12e030`, WebHarbor revision `381dbd98…`), drawn with `random.Random(118)` from
the 45 Allrecipes tasks. Allrecipes is the site the live demo (`s1a run allrecipes`) already runs, and one site
means one per-site image (`wildawm/webharbor:allrecipes-v1`, port 40000). Only ids and revisions are committed;
tasks, verifiers and images are downloaded on demand. Every task is labelled `state_change: unchecked` until the
spike reads its verifier; the site is reset before every attempt either way, so a task that writes cannot leak into
the next one.

```bash
uv run python -m evals.webharbor show                              # the ids and revisions
uv run python -m evals.webharbor select --registry registry.json   # re-derive the ids from the seed
uv run python -m evals.webharbor fetch --dest .external/webharbor  # task folders at the pinned revision
docker pull wildawm/webharbor:allrecipes-v1
uv run python -m evals.webharbor run --tasks-dir .external/webharbor --arms jev llm
uv run python -m evals.webharbor score evals/results/webharbor/<run>/attempts.jsonl
```

## One attempt

1. **Reset.** A fresh container of the per-site image per attempt (`docker rm -f`, `docker run` with the task's
   `WH_*` variables and the site port published on `127.0.0.1`), then wait for the start URL to answer. With
   `--control-url` (the all-sites image) the reset is `POST /reset/<site>` on the control plane, with
   `Authorization: Bearer $WEBSYN_CONTROL_TOKEN`. The verifier then runs inside that all-sites container,
   discovered as the one that publishes the control-plane host port, or named with `--control-container`.
   A failed reset (including a control plane that is not running in a visible container) is a failed
   attempt, not a skipped one. The default `s1a-webharbor` name is only for the fresh per-site path.
2. **Play.** The task's own `instruction.md` is the goal; the production `browse` front plays it with the arm's
   model (`jev` or another decision model, or `llm`, the chat model) under the same `--max-steps` and `--timeout`.
3. **Verify.** The final answer, a termination record and the final screenshot go into the container's
   `/logs/agent`, the task's `tests/` into `/tests`, and `tests/test.sh` runs inside the container as in Harbor's
   verifier phase. Success is `/logs/verifier/reward.json`'s `reward`; a missing or malformed reward is a verifier
   error, never a pass. No LLM judge decides success (WebHarbor's optional agent judge is not run here).

## The table

`results.md` names the dataset, the task revision and the image digest (`N/A` when Docker did not report one), and
per arm: passed / attempts with every attempt in the denominator (reset failures, agent errors, timeouts, empty
answers, verifier errors), a 95% Wilson interval, median episode time split into decision-model time and the rest
(`N/A` for the chat-model arm, which makes no decision calls), and the failure reasons. No numbers are committed
yet: a run needs Docker, the image and the models' keys.

## Open spike items

- Whether `build_trajectory.py`'s run-package gate accepts one final screenshot from our harness, or wants one per
  step (`screenshot_manifest.jsonl`'s exact shape is read from the WebHarbor harnesses, not yet checked live).
- Whether the per-site image serves the control plane too, or only the all-sites image does.
- Labelling each task read-only or writing from its verifier.

## Licence

At the time of writing the WebHarbor repository has no repo-root licence (GitHub reports none, and the
[FCMO newsletter](https://fcmo-ai.github.io/FCMO-AI-Newsletter/2026/11/03/webharbor-releases-153-local-web-agent-environments-and-3-518-tasks/)
noted it missing). So nothing from WebHarbor is vendored here: the slice is ids and revisions, and the tasks, the
verifiers and the images are fetched on demand by whoever runs it. Revisit when upstream adds a licence.
