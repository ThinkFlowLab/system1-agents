#!/usr/bin/env bash
# PR #36 evidence on a real server: s1a main at the #36 merge, Flights agent, --model jev pointed at system1-omni's Rust
# frontend (127.0.0.1:8080, omni-jev) in front of Open-Jev-9B (Open-Jev's reference server). Arms: the default 5 s
# deadline, S1A_DECISION_TIMEOUT_S=30, and S1A_DECISION_TIMEOUT_S=1 (a deadline the server exceeds). Frames per call.
set -u
K=$HOME/kit; MODEL=${MODEL:-jev-latest}; ARMS=${ARMS:-"default-5s timeout-30s timeout-1s"}
export PATH=$HOME/.local/bin:$K/node-v22.12.0-linux-x64/bin:$PATH
OUT=$K/results/timeout-$(date +%Y-%m-%d__%H-%M-%S); mkdir -p $OUT; echo "== out: $OUT"
[ -d $K/s1a-main ] || git clone -q https://github.com/ThinkFlowLab/system1-agents.git $K/s1a-main
cd $K/s1a-main && git fetch -q origin && git checkout -q 22e685b && cp $K/env.box .env && uv sync -q --extra dev
{ git log -1 --format="system1-agents %H (main, the #36 merge)"; git -C $K/system1-omni log -1 --format="system1-omni %H"; \
  git -C $K/Open-Jev log -1 --format="Open-Jev %H"; curl -s localhost:8080/health; echo; curl -s localhost:8791/health; echo; \
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; } | tee $OUT/versions.txt
C=$(ls -d $HOME/.cache/ms-playwright/chromium-*/chrome-linux*/chrome | head -1)
PY=$(uv run python -c "import sys;print(sys.executable)")
for arm in $ARMS; do
  pkill -f "[r]emote-debugging-port=9222"; sleep 2
  "$C" --headless=new --no-sandbox --remote-debugging-port=9222 --user-data-dir="$K/chrome-profile" --window-size=1280,900 \
    --no-first-run about:blank >/dev/null 2>&1 & sleep 5
  uv run --with websocket-client python $K/reject_cookies.py > /dev/null
  d=$OUT/$arm; mkdir -p $d; extra=()
  case $arm in timeout-30s) extra=(S1A_DECISION_TIMEOUT_S=30);; timeout-1s) extra=(S1A_DECISION_TIMEOUT_S=1);; esac
  echo "== $arm ${extra[*]:-} $(date -u +%H:%M:%S)" | tee -a $OUT/versions.txt
  env -u S1A_DECISION_TIMEOUT_S "${extra[@]}" TYPESAFE_API_KEY= TYPESAFE_API_URL=http://127.0.0.1:8080/v1/systemone TYPESAFE_MODEL=$MODEL \
    PLAYWRIGHT_MCP_COMMAND=$PY \
    PLAYWRIGHT_MCP_ARGS="-m evals.replay.cast --frames $d/frames -- npx -y @playwright/mcp@0.0.78 --cdp-endpoint=http://localhost:9222" \
    timeout 900 uv run s1a run flights --model jev --timeout 600 --logs-dir $d > $d/stdout.json 2> $d/stderr.log
  echo "   exit $? frames $(ls $d/frames 2>/dev/null | wc -l)" | tee -a $OUT/versions.txt
done
pkill -f "[r]emote-debugging-port=9222"
cd $K/results && tar czf $(basename $OUT).tar.gz $(basename $OUT) && echo "== TIMEOUT_DONE $(basename $OUT).tar.gz"
