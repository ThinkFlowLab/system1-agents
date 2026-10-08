# State, candidates, rules and budget

What the loop sends Jev on every decision: the observation, the candidates, the rules text.

## The observation (`observe`)

- Plain JSON of what a person would look at to decide: the pile, the board rows, the current question, the task.
- A `progress` key holding what "the same situation" means: the rethink rail compares it between steps to detect a
  stall. Histories and step counters change every step and belong outside `progress`.
- Short recent history when the decision depends on it (the last few actions and what they showed), clipped.
- Facts already derived that Jev should not compute: totals, "soft hand", the largest tile and its corner, places
  already checked and found empty.

## The candidates (`candidates`)

- Human-readable keys: `take_2`, `stand`, `go to shelf 1`, `UP`. Never bare indices.
- One line per key saying what the move does or leaves: "take 2, leaving 8 stones".
- Empty once the episode is done. Drop moves that did nothing last time (2048 does), and commands that never
  progress (ALFWorld drops `look`, `examine`, `inventory`).

## The rules text (`SPEC.rules`)

- Under about 120 words. Facts to recognise, phrased as what a good move looks like. Leave out chains of reasoning.
- Name the winning shapes and the traps in the state's own words so that Jev can match them by recognition.
- Every model reads the same text, including the chat-model arm.

## The budget and the baseline

- `max_steps`: the acts one episode may spend; the chat model gets twice as many iterations for malformed calls.
- `timeout_s`: the wall clock per episode; a timed-out episode keeps its score so far.
- `stall_after`: acts without a score change before the rethink rail asks the chat model for a plan; 0 for games
  where every act changes the score or the question.
- Give a `baseline` whenever a rule or an expert plan is known; it is the upper or lower bound the table shows.
- The `random` model draws from its own stream, derived from the episode seed. An env opponent seeded from the plain
  integer seed, as the template does, draws a different stream.
- In the summary a score above 0 counts as a win and below 0 as a loss. A draw scored 0.5 counts as a win there.

## Browser and rail specifics

- Browser agents keep the shipped operation rules until a site family shows a miss; then write that family's
  rules string in the spec. The goal is the task text; leave it `None` when every call brings its own.
- Rails send a few thousand characters of state at most: the tool name, the arguments or the result, clipped.
  Thresholds start at allow 0.3 and act 0.7; the labelled set decides where they move.
