# Screenshot target selection on macOS

The native test app draws Save and Cancel tiles without accessibility children. Reset swaps their positions.
The agent receives the current screenshot and chooses between two configured points. A successful click
shows Saved; the app also appends the selected tile to a temporary file for an independent check.

The optional `cua-four-b` extra loads the official Cua-S1 4B scorer locally. Set `CUA_S1_VARIANT=4b` and
`CUA_S1_MODALITY=multimodal` for screenshots, or `CUA_S1_MODALITY=text` for text observations. Existing
`--model cua` runs keep using Nano unless the variant is set. The 4B scorer supports up to 26 choices.

## Run the visual task

Requires macOS, `uv`, Swift command-line tools, and Cua Driver with Accessibility and Screen Recording
permissions. Keep the Mac unlocked. Close an older S1A visual fixture before rebuilding it.
Run these commands from the current project directory:

```bash
S1A_PROJECT_DIR="$(pwd -P)"
uv sync --project "$S1A_PROJECT_DIR" --extra cua-four-b
bash "$S1A_PROJECT_DIR/evals/desktop/build_visual_fixture.sh" /tmp/S1AVisualFixture.app
export CUA_DRIVER_BIN=/Applications/CuaDriver.app/Contents/MacOS/cua-driver
export CUA_S1_VARIANT=4b CUA_S1_MODALITY=multimodal
export CUA_S1_DEVICE=mps CUA_S1_DTYPE=bfloat16 HF_DEACTIVATE_ASYNC_LOAD=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
: > /tmp/s1a-visual-target.txt
uv run --project "$S1A_PROJECT_DIR" --no-sync s1a run desktop \
  --model cua --rethink off --episodes 4 --max-steps 1 \
  --app S1AVisualFixture --app-path /tmp/S1AVisualFixture.app --expect Saved \
  --window-title "S1A Visual Fixture" \
  --goal "Click the tile labelled Save in the screenshot" \
  --pixel-target left=0.27,0.51 --pixel-target right=0.73,0.51 \
  --clear Reset --execute
uv run --project "$S1A_PROJECT_DIR" --no-sync python -c \
  'from pathlib import Path; rows = Path("/tmp/s1a-visual-target.txt").read_text().splitlines(); assert rows == ["Save selected"] * 4, rows; print("Verified all four clicks")'
```

The base model and adapter download on first use. `CUA_S1_BASE_MODEL` and `CUA_S1_CHECKPOINT` can point
to downloaded directories. Omit `--execute` to preview a decision without clicking or resetting the app.

For a fixed-action check, use `--model rule --episodes 1 --plan pixel:right` on a newly launched fixture.
Its first Reset moves Save to the right; later resets alternate sides. `--pixel-target` coordinates are fractions
of the captured window. Every click carries its capture ID so the driver can reject a stale target.

The temporary selection file and ignored run outputs are generated locally. Screenshot bytes are passed
to the model separately from the serialized observation state.
