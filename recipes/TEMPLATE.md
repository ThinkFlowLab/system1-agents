# <Use case> with <agent>

## Summary

- Task and observable completion check:
- Agent / front (`tool`, `browser` or `rail`):
- State source and enumerated actions / question types:
- Decision model, checkpoint revision and inference engine:
- Chat-model role and credentials, if used:
- Environment / hardware:
- Validation status and evidence link:

Describe when this recipe is useful and what it covers. Use one complete tested profile as the starting point
when available; label instructions that have not been exercised. Keep model support, client integration and
end-to-end validation distinct.

## Prerequisites and setup

Give commands from a stated working directory. Specify source revisions, Python/dependencies, fixtures,
credentials by variable name, device/driver requirements and any application permissions. Separate downloads,
builds and model loading from inference. Link shared installation and configuration instructions.

For serving, identify the worker and frontend, show the readiness check, and pin System1-Omni as well as
System1-Agents. Verify that the worker's modality, question types, window and hardware match the client.
Describe a missing client/worker/configuration rather than inventing a supported command.

## Run the task

Give a complete copyable command with the task input or fixture, model, seed, step/time budgets and artifact
location. State any expected downloads or external calls. Provide a small baseline/control separately when it
helps explain mechanics. Keep chat planning/typed values and decision-model choices identifiable.

## Verify the result

Give a command or observable check that can fail when the task fails. Check the environment result, retained
errors and unprocessed work; `DONE`, exit code zero or one screenshot alone may not establish completion.
Show the expected output separately from the actual recorded outcome, with links to the raw artifacts.

## Demo and validation

For important PRs, link the required video of application/task + System1-Agents decision-model agent +
System1-Omni inference in the same run, using the [video guide](../CONTRIBUTING.md#agent-video-demos) and its
PR #35 example. A terminal video is valid for text tasks. Logs and rule controls complement the required clip.
State source/engine/checkpoint revisions, exact commands, inputs, environment and local changes. Identify
actual runs, scripted controls and replays, including cuts/speed and human intervention. Report missing media
or execution evidence as a review gap; keep the important PR draft until supplied or a maintainer accepts
the documented exception. Do not turn one successful episode into an accuracy or performance claim.

## Troubleshooting and limits

List concrete symptoms, their likely causes and the next check. Include platform constraints, required keys,
unsupported model/front combinations and incomplete integrations. State task/dataset limitations and known
failure behavior. Clean up only processes, fixtures and reservations owned by the recipe run.

## References

Link the agent implementation, shared configuration, evaluation protocol, model/engine documentation and
evidence behind any claims. Adjust relative links when copying this template into a use-case directory.
