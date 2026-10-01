# Served Laya fixtures

Responses recorded from running servers on 2026-10-01, one JSON file per case: `status`,
`content_type`, `body` and, where it was sent, the `request`.

| prefix | server |
|---|---|
| `worker.` | system1-omni's Laya worker at `3d6cb57` (PR #30): `python -m frontend.laya_mps --device mps --model english --require-device --compile --weights fp16`, `LAYA_API_KEY=fixture-token` |
| `frontend.` | the same worker behind `omni-jev` (system1-omni#2) |
| `frontend-down.` | `omni-jev` with the worker stopped |
| `laya-serve.` | plain `laya-serve`, no API key, MPS |

laya 0.3.20, torch 2.14.0, checkpoint `convaiinnovations/laya` at `55cf4c4`, M1 Pro.
`tests/test_served_laya_fixtures.py` checks them against `docs/api/laya-systemone.current.openapi.yaml`;
`tests/test_decision_models_served.py` replays them through a mock transport.

To record again, start a server and run `python capture.py <out-dir> <prefix> <port> [token]`.
