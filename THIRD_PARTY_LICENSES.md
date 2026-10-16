# Third-party licenses

## browser-use/jev-ultrafast

Operation rules in `s1a/browser/prompts.py` are adapted from
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast).

MIT License

Copyright (c) 2026 Browser Use

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## WebVoyager

The task text in `s1a/agents/allrecipes.py` (`TASK`, task `Allrecipes--0`) is from the WebVoyager task set,
[MinorJerry/WebVoyager](https://github.com/MinorJerry/WebVoyager) (`data/WebVoyager_data.jsonl`), described in
He et al., 2024, [WebVoyager: Building an End-to-End Web Agent with Large Multimodal
Models](https://arxiv.org/abs/2401.13919). Apache License 2.0; the full license text is this repository's `LICENSE`.

## 2048

`evals/2048` is 2048 by Gabriele Cirulli, MIT License. The full text is in `evals/2048/LICENSE.txt`.
Fonts under `evals/2048/style/fonts` are Clear Sans by Intel Corporation, Apache License 2.0.

## Open Trivia Database

`evals/millionaire/questions.json`, fetched on the first Millionaire run, holds questions from the
[Open Trivia Database](https://opentdb.com), Creative Commons Attribution-ShareAlike 4.0 International.

## InjecAgent

The records of `evals/labelled/injection-public.jsonl` whose note starts with `injecagent-` are built by
`scripts/build_injection_dataset.py` from [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) at
commit `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`: `data/test_cases_dh_base.json`, `data/test_cases_ds_base.json` and
`data/attacker_simulated_responses.json`.

MIT License

Copyright (c) 2023 Qiusi Zhan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## AgentDojo

The records of `evals/labelled/injection-public.jsonl` whose note starts with `agentdojo` are tool outputs of the
[ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) suites, benchmark version v1.2.2 of the `agentdojo`
0.1.35 package, produced by `scripts/build_injection_dataset.py` from the user tasks' ground truth with and without the
package's attacks, and from the suites' own read tools on the default environments.

MIT License

Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
