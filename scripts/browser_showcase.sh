#!/usr/bin/env bash
# One browser agent on both models, headed, with a frame after every browser call from the run's own Playwright MCP
# session (evals/replay/cast.py), then the pair GIFs from the median run of each model: replay.gif (the replay page)
# and strip.gif (the raw frames under a header band).
#
#   scripts/browser_showcase.sh allrecipes [RUNS]          # RUNS runs per model, default 1; ROOT= renders an existing root with RUNS=0
#   PYTHON=/path/to/python scripts/browser_showcase.sh ...  # an interpreter with the report extra
#
# Needs the Jev key and the chat-model key in .env, Node for the MCP server, and a screen: the sites these agents
# visit answer a headless Chromium with a bot wall.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON=${PYTHON:-.venv/bin/python}
agent=${1:?usage: scripts/browser_showcase.sh <agent> [RUNS]}
runs=${2:-1}
root=${ROOT:-runs/browser/$agent/showcase-$(date +%Y-%m-%d__%H-%M-%S)}
speed=${SPEED:-8}

eval "$(scripts/pw_mcp_nosettle.sh)"  # the MCP copy without the settle sleeps: PLAYWRIGHT_MCP_COMMAND and _ARGS
server="$PLAYWRIGHT_MCP_COMMAND $PLAYWRIGHT_MCP_ARGS"
for ((i = 1; i <= runs; i++)); do  # not seq: BSD seq counts down from 1 to 0
  for model in jev llm; do
    logs=$root/$model-$i
    mkdir -p "$logs"
    batch=off; [ "$model" = jev ] && batch=on
    echo "=== $agent $model run $i -> $logs" >&2
    PLAYWRIGHT_MCP_COMMAND=$PYTHON PLAYWRIGHT_MCP_ARGS="-m evals.replay.cast --frames $logs/frames -- $server" \
      "$PYTHON" -m s1a run "$agent" --model "$model" --batch "$batch" --headed --logs-dir "$logs" > "$logs/stdout.json"
  done
done

median() {  # the model's run with the median wall clock; the lower middle for an even count
  "$PYTHON" - "$root" "$1" <<'PY'
import json, sys
from pathlib import Path
root, model = Path(sys.argv[1]), sys.argv[2]
runs = sorted(root.glob(f"{model}-*"), key=lambda d: json.loads((d / "answer.json").read_text())["elapsed_ms"])
print(runs[(len(runs) - 1) // 2])
PY
}
"$PYTHON" -m evals.replay "$(median jev)" "$(median llm)" --out "$root/replay" --gif --strip --speed "$speed" --width 1600
