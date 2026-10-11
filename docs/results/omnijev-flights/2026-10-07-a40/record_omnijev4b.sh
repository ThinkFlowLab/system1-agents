#!/usr/bin/env bash
# PR #33 video: OmniJev-4B v1.1 Flights runs on the A40, a frame after every browser call (evals/replay/cast.py).
set -u
K=$HOME/kit; N=${N:-3}
export PATH=$HOME/.local/bin:$K/node-v22.12.0-linux-x64/bin:$PATH HF_HUB_OFFLINE=1
OUT=$K/results/omnijev4b-$(date +%Y-%m-%d__%H-%M-%S); mkdir -p $OUT; echo "== out: $OUT"
cd $K/system1-agents && cp $K/env.box .env && git fetch -q origin && git checkout -q omnijev && git pull -q --ff-only origin omnijev
{ git log -1 --format="system1-agents omnijev %H"; git -C $K/OmniJev log -1 --format="OmniJev %H"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -sr; } | tee $OUT/versions.txt
C=$(ls -d $HOME/.cache/ms-playwright/chromium-*/chrome-linux*/chrome | head -1); echo "chrome $C" >> $OUT/versions.txt
pkill -f "[r]emote-debugging-port=9222"; sleep 2
"$C" --headless=new --no-sandbox --remote-debugging-port=9222 --user-data-dir="$K/chrome-profile" --window-size=1280,900 \
  --no-first-run about:blank >/dev/null 2>&1 & sleep 5
uv run --with websocket-client python $K/reject_cookies.py | tee -a $OUT/versions.txt
PY=$(uv run python -c "import sys;print(sys.executable)")
for i in $(seq 1 $N); do
  d=$OUT/run$i; mkdir -p $d
  echo "== run $i $(date -u +%H:%M:%S)"
  OMNIJEV_REPO=$K/OmniJev OMNIJEV_CHECKPOINT=$K/ckpt-4b OMNIJEV_BASE=$K/base-4b PLAYWRIGHT_MCP_COMMAND=$PY \
  PLAYWRIGHT_MCP_ARGS="-m evals.replay.cast --frames $d/frames -- npx -y @playwright/mcp@0.0.78 --cdp-endpoint=http://localhost:9222" \
    timeout 900 uv run s1a run flights --model omnijev --timeout 600 --logs-dir $d > $d/stdout.json 2> $d/stderr.log
  echo "   exit $? frames $(ls $d/frames 2>/dev/null | wc -l) $(head -c 200 $d/stdout.json)"
done
pkill -f "[r]emote-debugging-port=9222"
cd $K/results && tar czf $(basename $OUT).tar.gz $(basename $OUT) && echo "== RECORD_DONE $(basename $OUT).tar.gz"
