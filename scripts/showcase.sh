#!/usr/bin/env bash
# One showcase episode per eval and model, same seed for both models, headless, frames on for the browser games;
# then the pair page under evals/showcase/replays/<eval>/ (with its frames) and the GIF alone under
# docs/results/<eval>/showcase/replay.gif. Never feeds the matrix: every job lands under evals/showcase/, which
# evals.table does not read.
#
#   scripts/showcase.sh [SEED] [EVALS...]        default seed 0, evals: blackjack game2048 millionaire alfworld
#   PYTHON=/path/to/python scripts/showcase.sh   an interpreter with the report extra and the games' extras; ALFWorld's
#                                                scene needs alfworld-visual (evals/replay/thor_replay.py), or THOR_PYTHON
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON=${PYTHON:-.venv/bin/python}
THOR_PYTHON=${THOR_PYTHON:-$PYTHON}
seed=${1:-0}; shift || true
evals=("$@"); [ ${#evals[@]} -eq 0 ] && evals=(blackjack game2048 millionaire alfworld)

run() {  # eval, model, extra args...: prints the trial folder; the run's last stdout line is its summary JSON
  local eval=$1 model=$2; shift 2
  local job
  job=$("$PYTHON" -m s1a run "$eval" --model "$model" --episodes 1 --seed "$seed" --showcase "$@" | tail -1 \
    | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["job_dir"])')
  ls -d "$job"/*--*/ | head -1
}

for eval in "${evals[@]}"; do
  case $eval in
    blackjack)   args=(--rethink off) ;;
    game2048)    args=(--rethink on --max-steps 150) ;;
    millionaire) args=(--rethink off) ;;
    alfworld)    args=(--rethink on --offset "$seed" --stride 1) ;;   # --seed is ignored; the game is file $seed
    *) echo "unknown eval $eval" >&2; exit 1 ;;
  esac
  echo "=== $eval jev"; jev=$(run "$eval" jev "${args[@]}")
  echo "=== $eval llm"; llm=$(run "$eval" llm "${args[@]}")
  if [ "$eval" = alfworld ]; then  # the embodied scene behind the text game, one frame per step, into agent/frames/
    if "$THOR_PYTHON" -c "import ai2thor" 2>/dev/null; then
      for trial in "$jev" "$llm"; do "$THOR_PYTHON" evals/replay/thor_replay.py "$trial"; done
    else
      echo "alfworld: no ai2thor in $THOR_PYTHON, the GIF holds the transcript alone (uv sync --extra alfworld-visual)" >&2
    fi
  fi
  out="evals/showcase/replays/$eval"
  "$PYTHON" -m evals.replay "$jev" "$llm" --out "$out" --gif --mode time --speed 4 --width 720
  docs="docs/results/${eval/game2048/2048}/showcase"  # docs/benchmarks.md embeds docs/results/2048/
  mkdir -p "$docs"
  cp "$out/replay.gif" "$docs/replay.gif"
done
