#!/usr/bin/env bash
# The core install, no keys, no network: `list`, `--help` for every agent, `decide` without a key, the MCP listing.
# A game whose extra is missing must fail with one line on stderr naming the extra; with the extra installed
# its help must work. CI runs this on `uv sync --extra dev` alone.
#
#   scripts/smoke.sh                       # through uv run
#   S1A=.venv/bin/s1a PY=.venv/bin/python scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
S1A=${S1A:-uv run --no-sync s1a}
PY=${PY:-uv run --no-sync python}
AGENTS="alfworld allrecipes blackjack desktop flights game2048 injection_guard millionaire ticket_router"

fail() { echo "smoke: $*" >&2; exit 1; }

# Native Windows Python emits CRLF even when launched from Git Bash.
listed=$($S1A list | tr -d '\r')
[ "$(echo "$listed" | tr '\n' ' ' | sed 's/ $//')" = "$AGENTS" ] || fail "list printed: $listed"
echo "list: $AGENTS"

for agent in allrecipes desktop flights game2048 injection_guard millionaire ticket_router; do
  $S1A run "$agent" --help >/dev/null || fail "run $agent --help failed"
  echo "help: $agent"
done

for agent in blackjack alfworld; do
  package=$([ "$agent" = blackjack ] && echo rlcard || echo alfworld)
  if $PY -c "import $package" 2>/dev/null; then
    $S1A run "$agent" --help >/dev/null || fail "run $agent --help failed with $package installed"
    echo "help: $agent ($package installed)"
  else
    set +e; err=$($S1A run "$agent" --help 2>&1 >/dev/null); code=$?; set -e
    [ "$code" -eq 1 ] || fail "run $agent --help exited $code without $package, expected 1"
    echo "$err" | grep -q "uv sync --extra $agent" || fail "run $agent --help said: $err"
    echo "help: $agent -> $err"
  fi
done

if $PY -c "import ai2thor" 2>/dev/null; then
  $PY evals/replay/thor_replay.py --help >/dev/null || fail "thor_replay --help failed with ai2thor installed"
  echo "help: thor_replay (ai2thor installed)"
else
  set +e; err=$($PY evals/replay/thor_replay.py --help 2>&1 >/dev/null); code=$?; set -e
  [ "$code" -eq 1 ] || fail "thor_replay --help exited $code without ai2thor, expected 1"
  echo "$err" | grep -q "uv sync --extra alfworld-visual" || fail "thor_replay --help said: $err"
  echo "help: thor_replay -> $err"
fi

set +e
err=$(TYPESAFE_API_KEY= OPENROUTER_API_KEY= $S1A decide --state '{}' --option a=one --option b=two --rules none 2>&1 >/dev/null)
code=$?
set -e
[ "$code" -eq 1 ] || fail "decide without a key exited $code, expected 1"
[ -n "$err" ] && [ "$(echo "$err" | wc -l)" -eq 1 ] || fail "decide without a key must write exactly one line, wrote: $err"
echo "decide without a key -> $err"

$PY - <<'EOF'
import asyncio, json
import s1a.entry  # noqa: F401  routes the harness logs to files, as the real entry does
from mcp.shared.memory import create_connected_server_and_client_session
from s1a import mcp_server

async def main() -> None:
    async with create_connected_server_and_client_session(mcp_server.server) as session:
        result = await session.call_tool("list_agents", {})
    rows = result.structuredContent["result"] if result.structuredContent else json.loads(result.content[0].text)
    assert not result.isError and len(rows) == 9, rows
    for row in rows:
        print(f"mcp list_agents: {row['name']} ({row['front']})")

asyncio.run(main())
EOF
echo "smoke: ok"
