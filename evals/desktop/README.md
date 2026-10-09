# Desktop text input on macOS

The native fixture has a Body field and Save/Clear buttons. Save writes the current text to
`/tmp/s1a-desktop-fixture.txt`. The agent checks both the visible result and the saved file.

Install Cua Driver and grant macOS Accessibility and Screen Recording permissions. Keep the Mac unlocked
and avoid interacting with the test window during a run.

## Local Laya

```bash
bash evals/desktop/run_local.sh --episodes 3
```

This builds the fixture if missing and uses local Laya to choose the input and Save actions. Set `LAYA_MODEL`
to a downloaded checkpoint directory, `S1A` to an existing CLI executable, or `CUA_DRIVER_BIN` to the installed
driver. The script's default command is `uv run --extra laya s1a`. No Jev or chat-model key is needed.

## Fixed-plan execution check

```bash
bash evals/desktop/build_fixture.sh /tmp/S1ADocumentFixture.app
uv run s1a run desktop --model rule --rethink off --episodes 1 \
  --app S1ADocumentFixture --app-path /tmp/S1ADocumentFixture.app \
  --window-title "S1A Document Fixture" --goal "Enter the text in Body and save it" --expect Saved \
  --text "本地输入测试" --text-target Body --text-mode replace \
  --verify-file /tmp/s1a-desktop-fixture.txt --clear Clear --plan 'type:Body,Save' --execute
```

Omit `--execute` to preview one decision. Close the fixture before rebuilding it. Run outputs stay in the
ignored `evals/results/` directory; they are not source files.
