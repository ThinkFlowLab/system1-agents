# Sokoban (text-board adaptation)

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
sample. They retained successful 2B RLCD cases before filling the set. Results here use
symbolic text boards and must not be compared as equivalent to Valen's image-only evaluation.
No learned-model performance has been measured by this import.

The simulator comes from Valen, not vllm-jev's demo assets. The level definitions are
bundled without changes; evaluator-only reference solutions live under `tests/fixtures`
and are never loaded by the agent. Tests replay all 100 reference solutions to validate
transitions and level compatibility; this is not model accuracy.

See [third-party provenance](../../THIRD_PARTY_LICENSES.md#valen-sokoban) for pinned
code/data revisions and [the upstream dataset card](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game)
for the original protocol and selection details.
