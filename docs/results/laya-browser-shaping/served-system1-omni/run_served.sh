#!/usr/bin/env bash
# system1-omni's Laya text worker (recipe/laya, CPU) behind its Rust frontend, then served_trial.py on the shaped
# requests. Run from a system1-omni checkout with the recipe's .venv-laya and a release build of omni-jev.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LAYA_HOST=127.0.0.1 LAYA_PORT=8710 LAYA_DEVICE=cpu LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
  .venv-laya/bin/laya-serve > laya_serve.log 2>&1 &
until curl -s 127.0.0.1:8710/health >/dev/null; do sleep 5; done
OMNI_JEV_BIND=127.0.0.1:8711 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8710 ./target/release/omni-jev > omni_jev.log 2>&1 &
until curl -s 127.0.0.1:8711/health >/dev/null; do sleep 2; done
curl -s 127.0.0.1:8711/health; echo
git log -1 --format="system1-omni %H %cd"
.venv-laya/bin/python "$HERE/served_trial.py" http://127.0.0.1:8711 "$HERE/requests.jsonl" "$HERE/served_results.jsonl"
pkill -f "[l]aya-serve"; pkill -f "[o]mni-jev"
