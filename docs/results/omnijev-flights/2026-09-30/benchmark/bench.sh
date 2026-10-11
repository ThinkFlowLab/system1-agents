#!/usr/bin/env bash
# Google Flights benchmark: success rate (vision judge on the final screenshot) and time, per decision model.
# Usage: bash bench.sh "jev omnijev clm" 5      (models, runs per model); results in results/bench.csv
W=~/chaimae; K=$W/gpu_kit; E=$W/clm-env
MODELS=${1:-"jev omnijev clm"}; RUNS=${2:-5}
export PATH=$HOME/.local/bin:$W/node-v22.12.0-linux-x64/bin:$PATH
export LD_LIBRARY_PATH=$W/libs/usr/lib/x86_64-linux-gnu
export PLAYWRIGHT_MCP_ARGS="-y @playwright/mcp@0.0.78 --cdp-endpoint=http://localhost:9222" HF_HUB_OFFLINE=1
mkdir -p $K/results
CSV=$K/results/bench2.csv
[ -s $CSV ] || echo "time,model,run,success,total_s,decisions,median_decision_ms,model_s,page_s,browser_s,reason" > $CSV

# CLM needs its two servers (Qwen3-8B embeddings + clm-serve); started once, left running
if [[ " $MODELS " == *" clm "* ]]; then
  if [ ! -x $E/bin/clm-serve ]; then echo "== installing vLLM + CLM (~5 min)"; uv venv -q --python 3.12 $E && uv pip install -q --python $E/bin/python vllm contrastive-lm || exit 1; fi
  curl -s localhost:8090/v1/models >/dev/null || { echo "== starting Qwen3-8B (download ~16 GB the first time)"; HF_HUB_OFFLINE=0 nohup $E/bin/vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --gpu-memory-utilization 0.5 --port 8090 > $K/results/vllm.log 2>&1 & }
  until curl -s localhost:8090/v1/models >/dev/null; do sleep 15; echo "   waiting for Qwen3-8B..."; done
  curl -s localhost:8700/health >/dev/null || { echo "== starting CLM"; HF_HUB_OFFLINE=0 nohup $E/bin/clm-serve --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings > $K/results/clm.log 2>&1 & }
  until curl -s localhost:8700/health >/dev/null; do sleep 5; echo "   waiting for CLM..."; done
fi

# CLM with a window that holds the whole page (8,192 tokens instead of 2,048): both servers restarted with it
if [[ " $MODELS " == *" clm8k "* ]]; then
  pkill -f "vllm serve" ; pkill -f clm-serve ; pkill -f "qev serve" ; sleep 5
  [ -x $E/bin/clm-serve ] || { echo "== installing vLLM + CLM (~5 min)"; uv venv -q --python 3.12 $E && uv pip install -q --python $E/bin/python vllm contrastive-lm || exit 1; }
  echo "== starting Qwen3-8B with an 8,192-token window"
  HF_HUB_OFFLINE=0 nohup $E/bin/vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 8192 --gpu-memory-utilization 0.5 --port 8090 > $K/results/vllm8k.log 2>&1 &
  until curl -s localhost:8090/v1/models >/dev/null; do sleep 15; echo "   waiting for Qwen3-8B..."; done
  echo "== starting CLM reading up to 8,192 tokens"
  HF_HUB_OFFLINE=0 nohup $E/bin/clm-serve --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings --max-tokens 8192 > $K/results/clm8k.log 2>&1 &
  until curl -s localhost:8700/health >/dev/null; do sleep 5; echo "   waiting for CLM..."; done
fi

# OneJev (27B or 9B): served by its own qev server on port 8000, which speaks Jev's API; the GPU cannot hold it next
# to CLM's servers, so those are stopped first. Only one OneJev size per call.
Q=$W/qev-env
for size in 27B 9B 4B 0.8B; do
  if [[ " $MODELS " == *" onejev$size "* || " $MODELS " == *" onejev${size}c "* ]]; then
    pkill -f "vllm serve" ; pkill -f clm-serve ; pkill -f "qev serve" ; sleep 5
    if [ ! -x $Q/bin/qev ]; then echo "== installing OneJev's server (qev)"; uv venv -q --python 3.12 $Q && uv pip install -q --python $Q/bin/python "qev[torch] @ git+https://github.com/OmniJev/OneJev.git" || exit 1; fi
    echo "== starting OneJev-$size (download the first time: 55 GB for 27B, 19 GB for 9B)"
    HF_HUB_OFFLINE=0 nohup $Q/bin/qev serve --model OmniJev/OneJev-$size --multimodal --port 8000 > $K/results/onejev_$size.log 2>&1 &
    until curl -s localhost:8000/health >/dev/null; do sleep 15; echo "   waiting for OneJev-$size... ($(tail -c 100 $K/results/onejev_$size.log | tr '\n' ' '))"; done
    curl -s localhost:8000/health; echo
  fi
done

# One headless Chrome with its own profile, cookies refused once
C=$(ls -d $HOME/.cache/ms-playwright/chromium-*/chrome-linux*/chrome | head -1)
pgrep -f "remote-debugging-port=9222" >/dev/null || { "$C" --headless=new --no-sandbox --remote-debugging-port=9222 --user-data-dir="$K/chrome-profile" --window-size=1280,900 --no-first-run about:blank >/dev/null 2>&1 & sleep 5; }
cd $K/system1-agents
uv run --with websocket-client python $K/reject_cookies.py >/dev/null
TASK=$(uv run python -c "from s1a.agents.flights import SPEC; print(SPEC.goal)" 2>/dev/null | tail -1)
echo "== task: $TASK"

for model in $MODELS; do
  for i in $(seq 1 $RUNS); do
    log=$K/results/bench_${model}_$(date +%H%M%S).log
    prof=${log%.log}.json  # the team's profiler: where the seconds go (model, page, browser)
    case $model in
      jev)     env -u TYPESAFE_API_URL -u TYPESAFE_MODEL uv run s1a run flights --model jev --timeout 600 --profile-out "$prof" > "$log" 2>&1 ;;
      clm|clm8k) TYPESAFE_API_URL=http://127.0.0.1:8700/v1/systemone TYPESAFE_MODEL=clm-latest uv run python $K/slow_ok.py run flights --model jev --timeout 600 --profile-out "$prof" > "$log" 2>&1 ;;
      onejev27B|onejev9B|onejev4B|onejev0.8B)TYPESAFE_API_URL=http://127.0.0.1:8000/v1/systemone TYPESAFE_MODEL=jev-latest uv run python $K/slow_ok.py run flights --model jev --timeout 600 --profile-out "$prof" > "$log" 2>&1 ;;
      onejev27Bc|onejev9Bc) TYPESAFE_API_URL=http://127.0.0.1:8000/v1/systemone TYPESAFE_MODEL=jev-latest uv run python $K/compact_ok.py run flights --model jev --timeout 600 --profile-out "$prof" > "$log" 2>&1 ;;
      omnijev) OMNIJEV_REPO=$K/OmniJev OMNIJEV_CHECKPOINT=$K/ckpt-4b OMNIJEV_BASE=$K/base-4b uv run s1a run flights --model omnijev --timeout 600 --profile-out "$prof" > "$log" 2>&1 ;;
    esac
    shot=${log%.log}.png
    uv run --with websocket-client python $K/final_shot.py "$shot" >/dev/null 2>&1
    verdict=$(uv run python $K/judge.py "$shot" "$TASK" 2>/dev/null | tail -1)
    line=$(uv run python - "$log" "$verdict" "$model" "$i" <<'PY'
import json, sys, time
log, verdict, model, run = sys.argv[1:5]
text = open(log, encoding="utf-8", errors="replace").read()
k = text.rfind('{"ok"')
try:
    answer = json.loads(text[k:text.index("\n", k)] if "\n" in text[k:] else text[k:]) if k >= 0 else {}
except json.JSONDecodeError:
    answer = {}
report = answer.get("report") or {}
try:
    judged = json.loads(verdict)
except json.JSONDecodeError:
    judged = {"success": False, "reason": "judge failed"}
reason = str(judged.get("reason", "")).replace(",", ";").replace("\n", " ")
try:  # the team's profiler (docs/benchmarks.md): decision = waiting on the model, probe = page, tool = browser
    phases = json.load(open(log[:-4] + ".json", encoding="utf-8")).get("phases") or {}
except (OSError, ValueError):
    phases = {}
model_s = phases.get("decision", phases.get("jev", 0)) / 1000
print(f"{time.strftime('%H:%M:%S')},{model},{run},{int(bool(judged.get('success')))},"
      f"{report.get('elapsed_ms', 0) / 1000:.1f},{report.get('decisions', 0)},{report.get('median_decision_ms', 0)},"
      f"{model_s:.1f},{phases.get('probe', 0) / 1000:.1f},{phases.get('tool', 0) / 1000:.1f},{reason}")
PY
)
    echo "$line" >> $CSV
    echo "$line"
    # the actions this run took, in order, to see where it goes wrong
    uv run python - "$log" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
k = text.rfind('{"ok"')
try:
    history = (json.loads(text[k:].split("\n", 1)[0]).get("report") or {}).get("history") or []
except (json.JSONDecodeError, ValueError):
    history = []
steps = [(h.get("text") and f"type {h['text']}") or f"{h.get('kind')} {str(h.get('action'))[:30]}" for h in history]
print("   actions: " + " > ".join(steps[:16]) + (" > ..." if len(steps) > 16 else ""))
PY
  done
done

echo "== summary (all rows in results/bench2.csv)"
uv run python - "$CSV" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
print(f"{'model':<10} {'runs':>4} {'success':>8} {'rate':>6} {'total s':>8} {'model s':>8} {'page s':>7} {'browser s':>10} {'per decision ms':>16}")
for model in dict.fromkeys(r["model"] for r in rows):
    rs = [r for r in rows if r["model"] == model]
    ok = sum(int(r["success"]) for r in rs)
    med = lambda key: statistics.median(float(r[key]) for r in rs)
    print(f"{model:<10} {len(rs):>4} {ok:>8} {100 * ok / len(rs):>5.0f}% {med('total_s'):>8.1f} {med('model_s'):>8.1f} "
          f"{med('page_s'):>7.1f} {med('browser_s'):>10.1f} {med('median_decision_ms'):>16.0f}")
PY
