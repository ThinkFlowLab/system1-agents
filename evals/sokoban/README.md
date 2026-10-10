# Sokoban (text boards and, with --visual, Valen's rendered boards)

```bash
uv run s1a run sokoban --model random --rethink off --episodes 1 --max-steps 10
uv run s1a run sokoban --model jev --rethink off --episodes 100
```

No additional dependency, browser, dataset download or GPU is needed for the environment.
The model receives an ASCII board and remaining step budget, with the legend in its rules.
The four directions remain available even when blocked; blocked moves consume a step.
There is no solver, legal-action mask, undo, rule baseline or rethink assistance.

`--seed` selects the zero-based level offset, with consecutive levels per episode; the
range must stay within the bundled 100 levels. The episode ends on success or after at
most 200 actions (`--max-steps` can lower this); `--timeout` also caps wall time.
Score is 1 for a solved board and 0 otherwise, so `mean_score` is the solved fraction
among non-error episodes. Always report errors separately. Episode metadata records the
level ID, text observation mode, pushes and whether the environment step limit was reached.
The standard episode records include the final board, steps, elapsed time and decisions.

These are Valen's **selected** 100 evaluation levels (35 easy, 65 medium), not an unbiased
sample. They retained successful 2B RLCD cases before filling the set. Text-mode results use
symbolic boards and must not be compared as equivalent to Valen's image-only evaluation.
No learned-model performance has been measured by this import.

## Visual mode

```bash
uv run --extra visual s1a run sokoban --model random --rethink off --episodes 1 --max-steps 10 --visual
```

The `visual` extra (Pillow) rides on the command; `uv sync --extra visual` alone would prune
the other extras, so add it to your usual sync set instead (`uv sync --extra dev --extra report --extra visual`).

`--visual` keeps the levels, episode loop, budgets and scoring untouched and swaps the
observation: each decision is shown Valen's own rendering of the board, one PNG per move,
drawn with the `theme` and `tile_size` the level carries, alongside the remaining step
budget. The ASCII board never accompanies the picture — the preview checkpoint read images —
and the pictures stay out of the episode records (`observation_mode: visual` marks the mode).
The rules and the four direction labels follow each level's `language` (`en`/`zh`) and are
Valen's own instruction texts, verbatim.

A text-only backend — `jev`, `clm`, `laya`, `laya-served`, `cua` in its text modalities, and `llm` (the chat
model reads tool text only) — is refused before the first decision rather than run blind over a board it
cannot see; **the jev command at the top of this page is the text-mode run**. The guard admits what can
actually see the board — `cua`'s multimodal 4B — but admission is capability, not a Sokoban
result: no visual run is recorded. (`omnijev` is image-capable too, but it is a browser-front choice;
`s1a run` cannot select it.) `--model random` is the other exception, the offline smoke for
the rendering; the served Valen worker behind `/v1/systemone` is the backend this observation was made for
and ships as its own change. Its
protocol takes one PNG/JPEG data URL as `state.image`, capped at an 8 MiB body, a 4 MiB image
and 2048 pixels per side — a bundled level at tile 44 renders far below all of these.

This is the observation Valen's preview checkpoint was trained and validated on; the text
mode above remains the off-distribution control. Measured results belong to the runs recorded
under `evals/results/`, not to this import.

The simulator and the renderer both come from Valen, not vllm-jev's demo assets. The level
definitions are bundled without changes; evaluator-only reference solutions live under
`tests/fixtures` and are never loaded by the agent. Tests replay reference solutions over
both observation modes to validate transitions, level compatibility and rendering; this is
not model accuracy.

See [third-party provenance](../../THIRD_PARTY_LICENSES.md#valen-sokoban) for pinned
code/data revisions and [the upstream dataset card](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game)
for the original protocol and selection details.
