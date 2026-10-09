#!/usr/bin/env bash
# Runs the native document demo with local Laya; no chat or Jev calls.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
app="${S1A_FIXTURE_APP:-/tmp/S1ADocumentFixture.app}"
if [[ ! -x "$app/Contents/MacOS/S1ADocumentFixture" ]]; then
  bash evals/desktop/build_fixture.sh "$app"
fi
if [[ -n "${S1A:-}" ]]; then
  s1a_command=("$S1A")
else
  s1a_command=(uv run --extra laya s1a)
fi
exec "${s1a_command[@]}" run desktop --model laya --rethink off --episodes 1 \
  --app S1ADocumentFixture --app-path "$app" --window-title "S1A Document Fixture" \
  --goal "Enter the task text in Body and save it" --expect Saved \
  --text "Local Laya agent demo" --text-target Body --text-mode replace --verify-file /tmp/s1a-desktop-fixture.txt \
  --clear Clear --execute --log "$@"
