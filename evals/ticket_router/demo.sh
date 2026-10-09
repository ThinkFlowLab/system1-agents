#!/bin/bash
# The demo CONTRIBUTING.md asks for: application/task -> system1-agents -> System1-Omni inference -> result,
# in one run, through the frontend rather than straight at clm-serve.
#
#   CLM_URL=http://127.0.0.1:8080 evals/ticket_router/demo.sh
#
# Assumes `clm-serve` is up behind `omni-jev` (see docs/clm.md) and a `uv` environment in the checkout.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY=.venv/bin/python
# `s1a run` prints an absolute job_dir; strip the checkout prefix so the transcript (and anything recorded from
# it) names paths relative to the repository rather than to whoever ran it.
QUIET="grep -v -e 'INFO |' -e 'SyntaxWarning' -e 'txt = ' -e 'for match in' | sed 's#$PWD/##g'"
CLM_URL=${CLM_URL:?set CLM_URL to the omni-jev frontend, e.g. http://127.0.0.1:8080}

echo "# application/task   ticket routing: 30 labelled tickets, five queues"
echo "# agents             s1a run ticket_router --model clm"
echo "# System1-Omni       omni-jev -> clm-serve -> Qwen3-8B on one RTX 4090"
echo

echo "\$ curl -s \$CLM_URL/health          # the frontend, ready, saying who is behind it"
curl -s "$CLM_URL/health" | $PY -c "
import json, sys
d = json.load(sys.stdin)
print('  ', json.dumps({k: d[k] for k in ('ok', 'embedder', 'models')}))
print('   vector cache on', d['cache']['device'])
" 2>&1 | grep -v "INFO |"
echo

echo "\$ three of the thirty labelled tickets this run routes (seed 0 routes all of them)"
$PY -c "
import random, sys
sys.path.insert(0, '.')
from s1a.agents.ticket_router import load_tickets, DEFAULT_DATASET
rows = load_tickets(DEFAULT_DATASET)
random.Random(0).shuffle(rows)
for r in rows[:3]:
    print(f\"   {r['id']}  label={r['label']:9} {r['title']}\")
    print(f\"      {r['description'][:86]}\")
" 2>&1 | grep -v "INFO |"
echo

echo "\$ CLM_URL=\$CLM_URL s1a run ticket_router --model clm --rethink off \\"
echo "      --episodes 1 --seed 0 --showcase --log"
uv run --no-sync s1a run ticket_router --model clm --rethink off \
  --episodes 1 --seed 0 --showcase --log 2>&1 | eval $QUIET
echo

echo "\$ the identity recorded in every tick, from the run's own episode.json"
JOB=$(ls -dt evals/showcase/ticket_router/*__clm | head -1)
$PY -c "
import glob, json
from collections import Counter
d = json.load(open(glob.glob('$JOB/*/agent/episode.json')[0]))
t = d['decisions'][0]
print('   source', t['source'], '| model', t['model'], '| ms', t['ms'])
print('   served_by', json.dumps(t['served_by']))
tr = d['extra']['ticket_router']
print('   answered', dict(Counter(r['predicted'] for r in tr['routes'])))
print('   correct ', str(tr['correct']) + '/' + str(tr['total']), 'routed as labelled')
" 2>&1 | grep -v "INFO |"
