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

## Valen Sokoban

`s1a/agents/_sokoban.py` is adapted from
[Liuziyu77/Valen](https://github.com/Liuziyu77/Valen/blob/c96c4f736c6928d2cb1166bc105a24cebdfcdc40/evaluation/sokoban/env.py),
commit `c96c4f736c6928d2cb1166bc105a24cebdfcdc40`, Apache-2.0.
Changes: removed unused hash, inverse-action and clone helpers; applied repository formatting.

`s1a/agents/_data/sokoban_levels.jsonl` and `tests/fixtures/sokoban_reference.jsonl` are unchanged copies of
`eval_sokoban/levels.jsonl` and `eval_sokoban/reference.jsonl` from
[Valen-Team/Valen-Eval-Game](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game/tree/4f25038cd30d9a96aa377eea176a4d2e1680a35c),
revision `4f25038cd30d9a96aa377eea176a4d2e1680a35c`, declared Apache-2.0.
The full Apache-2.0 license text is this repository's `LICENSE`.
