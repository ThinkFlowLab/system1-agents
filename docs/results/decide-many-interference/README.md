# decide_many cross-question interference

Placeholder for a real `s1a probe --bundle` run (#127). No live row is recorded in this PR.

Intended first run, when weights or a key are available:

```bash
uv run s1a probe evals/probe/interference.jsonl --model laya --bundle --json laya-bundle.json
```

Record here: the command, Agents SHA, checkpoint, device, date, and case count. Flip rates in that file would be this repository's, not the Tev / Clef percentages from arXiv:2611.12521.

This box did not have in-process Laya weights or a Jev key at the time the harness landed, so no numbers are invented.

See [docs/decide-many-interference.md](../../decide-many-interference.md) for the protocols, the committed cases, and the statement that ticket-router is single-question today while browser ticks are the in-repo bundled path.
