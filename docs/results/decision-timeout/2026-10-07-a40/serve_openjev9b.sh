#!/usr/bin/env bash
# Open-Jev-9B (Open-Jev's reference server, GPU) behind system1-omni's Rust frontend (omni-jev).
set -euo pipefail
K=$HOME/kit; export PATH=$HOME/.local/bin:$PATH
[ -d $K/oj-gpu ] || uv venv -q -p 3.12 $K/oj-gpu
uv pip install -q -p $K/oj-gpu/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -q -p $K/oj-gpu/bin/python transformers==5.10.2 peft==0.19.1 accelerate==1.13.0 safetensors "huggingface_hub[cli]" -e $K/Open-Jev
$K/oj-gpu/bin/python -c "import torch;print('torch',torch.__version__,torch.cuda.is_available())"
$K/oj-gpu/bin/hf download ZefanCai/Open-Jev-9B --revision 47e9668 --local-dir $K/Open-Jev-9B >/dev/null
$K/oj-gpu/bin/hf download Qwen/Qwen3.5-9B --revision c202236 >/dev/null
du -sh $K/Open-Jev-9B; ls $K/Open-Jev-9B $K/Open-Jev-9B/package 2>/dev/null | head -20
pkill -f "[j]ev.server" || true; pkill -f "[o]mni-jev" || true; sleep 2
cd $K/Open-Jev && setsid nohup $K/oj-gpu/bin/python -m jev.server --checkpoint $K/Open-Jev-9B/package/checkpoint --device cuda:0 --port 8791 \
  > $K/oj9b-server.log 2>&1 < /dev/null &
until curl -s localhost:8791/health; do sleep 10; grep -q Traceback $K/oj9b-server.log && { tail -30 $K/oj9b-server.log; exit 1; }; done; echo
cd $K/system1-omni && OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8791 setsid nohup target/release/omni-jev \
  > $K/omni-frontend.log 2>&1 < /dev/null &
sleep 3
curl -s -w '\nHTTP %{http_code} %{time_total}s\n' localhost:8080/v1/systemone -H 'Content-Type: application/json' \
  --data-binary @recipe/open_jev/example-request.json | tail -c 600
echo "== SERVE_READY"
