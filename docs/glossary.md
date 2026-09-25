# Glossary

Terms the documentation glosses on first mention, collected in one place for quick lookup.

- **System 1 decision model**: A fast classifier that reads environment state and returns probabilities over caller-enumerated options in one forward pass without generating freeform text. [docs/decision-models.md](decision-models.md)
- **Decision model**: The abstract interface (`DecisionModel`) standardizing how fronts query backends (`jev`, `laya`, `cua`, baselines) for typed decisions. [docs/decision-models.md](decision-models.md)
- **Model slot**: The constructor argument (`model`) of an openJiuwen agent where a decision model or chat model is plugged in to take actions on decision turns. [docs/architecture.md](architecture.md)
- **Front (tool, browser, rail)**: An agent interaction layer (`ToolDecisionModel`, `BrowserDecisionModel`, `DecisionModelRail`) that extracts state, presents candidate options to a decision model, and executes actions. [docs/architecture.md](architecture.md)
- **Rail**: A policy or safety check that evaluates state at lifecycle hooks to enforce constraints, block loops, or fail closed. [docs/architecture.md](architecture.md)
- **Head (`choice`, `noul`, `score`)**: A specialized prediction head of a decision model; `choice` selects an offered key, `noul` estimates proposition probability, and `score` ranks state against a rubric. [docs/decision-models.md](decision-models.md)
- **Arm**: An experimental configuration or driver pipeline tested on identical benchmark tasks to measure comparative latency, cost, and accuracy. [docs/benchmarks.md](benchmarks.md)
- **Tick**: A single decision turn in an agent episode recording the chosen key, probabilities, confidence, latency, tokens, and active state flags. [evals/README.md](../evals/README.md)
- **Replay**: A visual or step-by-step reconstruction of an evaluation run, rendering states, chosen actions, and model probabilities side by side to HTML or GIF. [evals/README.md](../evals/README.md)
- **Episode**: A single complete run of an agent through a task, environment, or game from start to terminal state or timeout. [evals/README.md](../evals/README.md)
- **Labelled set**: A dataset of pre-recorded environment states and ground-truth options in JSONL format used to benchmark precision, recall, and accuracy without live execution. [evals/README.md](../evals/README.md)
