#!/usr/bin/env bash
# PR #36 on system1-omni's native Open-Jev worker: Open-Jev-9B (supported on main since 4a79980), one A40.
set -euo pipefail
K=$HOME/kit; O=$K/system1-omni; export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH
pkill -f "[j]ev.server" || true; pkill -f "[o]mni-jev" || true; pkill -f "[o]mni-open-jev-native" || true; sleep 2
cd $O && git fetch -q origin && git checkout -q 4a79980 && git log -1 --format="system1-omni %H"
src/backends/cuda/qwen3_5/build.sh target/release 86
cargo build --release --locked -q -p omni-open-jev-native -p omni-jev
HF=$K/oj-env/bin/hf
$HF download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir weights/Qwen3.5-9B >/dev/null
$HF download ZefanCai/Open-Jev-9B --revision 47e966881e489511c0c7f5633a9e1960a676a551 --include 'package/checkpoint/*' --local-dir weights/Open-Jev-9B >/dev/null
[ -f weights/open-jev-9b-merged/open_jev_export.json ] || /usr/bin/time -v env CUDA_VISIBLE_DEVICES='' $K/oj-env/bin/python recipe/open_jev/export_merged.py \
  --base weights/Qwen3.5-9B --checkpoint weights/Open-Jev-9B/package/checkpoint --out weights/open-jev-9b-merged 2>&1 | grep -E "Elapsed|Maximum resident|Exit status"
du -sh weights/open-jev-9b-merged
setsid nohup env OPEN_JEV_MODEL=weights/open-jev-9b-merged target/release/omni-open-jev-native > $K/native9b-worker.log 2>&1 < /dev/null &
until curl -s localhost:8000/health; do sleep 10; grep -qi "error" $K/native9b-worker.log && { cat $K/native9b-worker.log; exit 1; }; done; echo
setsid nohup env OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 target/release/omni-jev > $K/native9b-frontend.log 2>&1 < /dev/null &
sleep 3; curl -s localhost:8080/health; echo
curl -s -w '\nHTTP %{http_code} %{time_total}s\n' localhost:8080/v1/systemone -H 'Content-Type: application/json' --data-binary @$O/recipe/open_jev/example-request.json | tail -c 400
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "== NATIVE_READY"
