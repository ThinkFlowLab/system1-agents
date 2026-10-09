# Verify screenshot target selection

Use the [visual fixture and reproduction commands](visual.md). Reset alternates Save and Cancel, and the
agent chooses between the two task-provided points using the current screenshot. Verify each actual click
against a fresh append to the fixture's output file. Keep the wrong-point control: clicking Cancel must
write `Cancel selected` and score 0.

When recording a trial, retain the model/checkpoint revisions, source commit, device and precision, model
choices, capture/window binding and independent file checks. Label fixed-point controls separately from
model-selected actions. A mocked Windows driver test is separate from native desktop validation.

[Historical screenshots, complete decision traces and reproduction records](https://github.com/QianCyrus/system1-agents/releases/tag/pr-review-evidence-20261007)
are preserved in the attached archive, with their original source bindings and negative control. These
one-off records and media are no longer maintained in the source tree; the fixture and regression tests remain.

The adapter in this PR calls Cua-S1 4B in process. The HTTP screenshot bridge and System1-Omni CUDA trial
are tracked in [#44](https://github.com/ThinkFlowLab/system1-agents/issues/44). In-process MPS evidence does
not establish that separate served path.
