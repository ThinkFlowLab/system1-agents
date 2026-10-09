# Verify desktop text input and Save

The maintained [native fixture](fixture.swift) writes `/tmp/s1a-desktop-fixture.txt` only when Save is
pressed. Clear empties Body and removes the output file. A successful task requires exact field readback,
the Saved status and a fresh file containing the task's supplied text.

## Served Laya trial

Start a compatible System1-Omni worker using the [served-Laya guide](../../docs/served-laya.md). Save its
`/health` response and retain each decision's `served_by` metadata when recording evidence. This path
sends text observations over HTTP; it does not require the in-process `laya` extra in the desktop client.

From this checkout, with macOS, Swift and Cua Driver permissions ready:

```bash
uv sync --extra dev --frozen
bash evals/desktop/build_fixture.sh /tmp/S1AServedDocument.app
export CUA_DRIVER_BIN=/Applications/CuaDriver.app/Contents/MacOS/cua-driver
export CUA_DRIVER_PERMISSION_MODE=standard
export LAYA_SERVED_URL=http://127.0.0.1:18127 LAYA_SERVED_MODEL=english
export DEMO_STARTED_NS="$(uv run --no-sync python -c 'import time; print(time.time_ns())')"
uv run --no-sync s1a run desktop \
  --model laya-served --rethink off --episodes 1 --seed 0 --max-steps 8 \
  --app S1ADocumentFixture --app-path /tmp/S1AServedDocument.app \
  --window-title 'S1A Document Fixture' \
  --goal 'Enter the task text in Body and save it' --expect Saved \
  --text 'Served Laya desktop demo' --text-target Body --text-mode replace \
  --verify-file /tmp/s1a-desktop-fixture.txt --clear Clear --execute --log
uv run --no-sync python - <<'PYTHON'
import os
from pathlib import Path

output = Path('/tmp/s1a-desktop-fixture.txt')
assert output.stat().st_mtime_ns >= int(os.environ['DEMO_STARTED_NS'])
assert output.read_text() == 'Served Laya desktop demo'
print('Fresh file matches the supplied text')
PYTHON
```

Keep all actions in the trace, including any premature Save. For a fixed-plan execution check, use
`--model rule --plan 'type:Body,Save'`; label it separately from model-selected actions. The existing
[desktop guide](README.md) also describes in-process Laya and Chinese multiline input.

## Review artifacts

[Video, complete served trace and historical evidence archive](https://github.com/QianCyrus/system1-agents/releases/tag/pr-review-evidence-20261007)
retain the original run/source bindings, ineffective actions, negative controls, warnings and reproduction
commands. One-off records and media are attached there rather than maintained as source files. The fixture,
regression tests and commands above remain in the repository.
